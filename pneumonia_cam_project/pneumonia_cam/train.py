import argparse
import torch
import torch.nn as nn # For nn.DataParallel
import torch.optim as optim
import pandas as pd
from pathlib import Path
import yaml # For config file
import time # For timing epochs
import os # For num_workers default
import sys # For sys.path manipulation if needed

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import torchmetrics

# Add project root to sys.path to allow direct import of pneumonia_cam components
# This is useful if the script is run directly, e.g., python pneumonia_cam/train.py
# Assumes the script is in pneumonia_cam_project/pneumonia_cam/
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pneumonia_cam.utils import get_device, format_time, set_seed, clear_memory
from pneumonia_cam.data_loader import create_dataloaders 
# Default transforms are not directly used by train.py, but by create_dataloaders if specific ones aren't passed.
# from pneumonia_cam.data_loader import default_train_transforms, default_val_transforms 
from pneumonia_cam.model import PneumoniaModelCAM

def define_arg_parser():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model with CAM.")
    
    # Config file
    parser.add_argument('--config_file', type=str, help='Path to YAML config file to override default args.')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=False, help='Path to processed data directory (containing train/val splits and dataset_stats.json).')
    parser.add_argument('--label_csv', type=str, required=False, help='Path to the CSV file with labels (e.g., stage_2_train_labels.csv).')
    parser.add_argument('--output_dir', type=str, default='./runs/pneumonia_detection_run', help='Directory to save checkpoints and logs.')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='Name of the TIMM pretrained model.')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes (1 for binary pneumonia detection).')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input image channels (1 for grayscale).')
    parser.add_argument('--img_size', type=int, default=224, help='Expected input image size (square).')
    parser.add_argument('--pos_weight', type=float, default=2.0, help='Positive class weight for BCEWithLogitsLoss. Set to None or 0 to disable.')
    parser.add_argument('--classifier_hidden_dim', type=int, default=512, help='Hidden dimension for the EnhancedClassifier.')
    parser.add_argument('--classifier_dropout_rate', type=float, default=0.3, help='Dropout rate for the EnhancedClassifier.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate for AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for DataLoader. Defaults to os.cpu_count().')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping (epochs). Set to 0 to disable.')
    parser.add_argument('--use_weighted_sampler', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use weighted sampler for training data (True/False).')


    # Will be filled by main() if required args are missing after config load
    parser.required_args_after_config = ['data_dir', 'label_csv'] 
    return parser

def main(args):
    # Setup
    set_seed(args.seed)
    output_path = Path(args.output_dir)
    logs_path = output_path / 'logs'
    output_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # Save config to output_dir
    with open(output_path / 'train_config.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    device = get_device()
    writer = SummaryWriter(log_dir=str(logs_path))
    
    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    pos_weight_actual = args.pos_weight if args.pos_weight is not None and args.pos_weight > 0 else None


    print("Starting training script with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"Device: {device}")
    print(f"Number of workers for DataLoader: {num_workers}")
    if pos_weight_actual is not None:
        print(f"Using positive class weight: {pos_weight_actual}")


    # Data
    print(f"Loading labels from: {args.label_csv}")
    if not Path(args.label_csv).exists():
        raise FileNotFoundError(f"Label CSV file not found: {args.label_csv}")
    labels_df = pd.read_csv(args.label_csv)

    if not labels_df.empty:
        relevant_columns = labels_df.columns.tolist() # Simplified, assuming relevant cols exist
        # Ensure 'patientId' and 'Target' are present
        if 'patientId' not in labels_df.columns or 'Target' not in labels_df.columns:
            raise ValueError("Label CSV must contain 'patientId' and 'Target' columns.")
            
        # For deduplication, we need to handle cases where some image-specific columns might be NaN for Target=0
        # A more robust way for deduplication focusing on one entry per patient:
        if labels_df['patientId'].duplicated().any():
            print("Deduplicating labels to ensure one entry per patientId...")
            # If 'nonnull_count' logic from notebook is crucial:
            # Need to ensure 'x', 'y', 'width', 'height' exist or are added with NaNs
            cols_for_nonnull = ['x', 'y', 'width', 'height', 'Target']
            for col in cols_for_nonnull:
                if col not in labels_df.columns: labels_df[col] = np.nan
            labels_df["nonnull_count"] = labels_df[cols_for_nonnull].notnull().sum(axis=1)
            max_nonnull_idx = labels_df.groupby("patientId")["nonnull_count"].idxmax()
            deduplicated_labels_df = labels_df.loc[max_nonnull_idx].copy()
            deduplicated_labels_df.drop(columns="nonnull_count", inplace=True, errors='ignore')
        else:
            deduplicated_labels_df = labels_df.copy()
        deduplicated_labels_df.reset_index(drop=True, inplace=True)
        print(f"Using {len(deduplicated_labels_df)} unique patient entries after deduplication.")
    else:
        raise ValueError("Label CSV file is empty or could not be read.")

    print(f"Creating DataLoaders for data in: {args.data_dir}")
    train_loader, val_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        label_df=deduplicated_labels_df, # Use the deduplicated dataframe
        batch_size=args.batch_size,
        num_workers=num_workers,
        use_weighted_sampler=args.use_weighted_sampler
        # Using default transforms from data_loader.py which PneumoniaDataset will apply
    )

    # Model, Optimizer, Criterion
    print(f"Initializing model: {args.model_name}")
    model = PneumoniaModelCAM(
        pretrained_model_name=args.model_name,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        img_size=args.img_size,
        criterion_pos_weight=pos_weight_actual,
        classifier_hidden_dim=args.classifier_hidden_dim,
        classifier_dropout_rate=args.classifier_dropout_rate
    )
    model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using nn.DataParallel for {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Criterion is part of the model instance (PneumoniaModelCAM)
    # If using DataParallel, access criterion from model.module
    criterion = model.module.criterion if isinstance(model, nn.DataParallel) else model.criterion
    # Ensure criterion's pos_weight (if any) is on the correct device
    if hasattr(criterion, 'pos_weight') and criterion.pos_weight is not None:
        criterion.pos_weight = criterion.pos_weight.to(device)


    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Metrics
    train_accuracy = torchmetrics.Accuracy(task="binary").to(device)
    train_auroc = torchmetrics.AUROC(task="binary").to(device)
    val_accuracy = torchmetrics.Accuracy(task="binary").to(device)
    val_auroc = torchmetrics.AUROC(task="binary").to(device)

    # Training Loop
    best_val_metric = 0.0 # Using AUROC for best model saving
    epochs_no_improve = 0
    overall_start_time = time.time()

    print(f"🚀 Starting training for {args.epochs} epochs on {device}.")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_accuracy.reset(); train_auroc.reset()
        
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False, unit="batch")
        for images, labels in train_progress_bar:
            images, labels = images.to(device), labels.to(device) # labels already [B,1] from Dataset

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # PneumoniaModelCAM returns (logits, features_map)
                # Logits are already squeezed if num_classes=1
                logits, _ = model(images) 
                loss = criterion(logits, labels.squeeze(-1)) # Squeeze label for BCEWithLogitsLoss if logits are [B]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * images.size(0)
            preds_proba = torch.sigmoid(logits.detach())
            train_accuracy.update(preds_proba, labels.squeeze(-1).int())
            train_auroc.update(preds_proba, labels.squeeze(-1).int())
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        epoch_train_acc = train_accuracy.compute()
        epoch_train_auc = train_auroc.compute()

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_train_acc, epoch)
        writer.add_scalar("AUROC/train_epoch", epoch_train_auc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_accuracy.reset(); val_auroc.reset()

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images, labels = images.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits, _ = model(images)
                    loss = criterion(logits, labels.squeeze(-1))
                
                running_val_loss += loss.item() * images.size(0)
                preds_proba = torch.sigmoid(logits.detach())
                val_accuracy.update(preds_proba, labels.squeeze(-1).int())
                val_auroc.update(preds_proba, labels.squeeze(-1).int())
                val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = running_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        epoch_val_acc = val_accuracy.compute()
        epoch_val_auc = val_auroc.compute()

        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val_epoch", epoch_val_acc, epoch)
        writer.add_scalar("AUROC/val_epoch", epoch_val_auc, epoch)

        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, AUC: {epoch_train_auc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, AUC: {epoch_val_auc:.4f} | "
              f"Time: {format_time(epoch_duration)}")

        # Checkpoint saving and Early stopping
        current_metric = epoch_val_auc # Using AUROC for checkpointing and early stopping
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            epochs_no_improve = 0
            checkpoint_path = output_path / f"best_model_epoch_{epoch+1}_auc_{best_val_metric:.4f}.pth"
            # Save model state, optimizer state, epoch, etc.
            save_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'args': vars(args) # Save arguments for reference
            }, checkpoint_path)
            print(f"🏅 Checkpoint saved: {checkpoint_path}")
        else:
            epochs_no_improve += 1

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"✋ Early stopping triggered after {epochs_no_improve} epochs without improvement on Val AUC.")
            break
            
    # Cleanup
    total_training_time = time.time() - overall_start_time
    print(f"--- 🏆 Training Finished --- Total Time: {format_time(total_training_time)} ---")
    print(f"Best Validation AUROC: {best_val_metric:.4f}")
    writer.close()
    clear_memory() # Optional: clear memory at the end


if __name__ == '__main__':
    parser = define_arg_parser()
    parsed_args = parser.parse_args()

    # Load from config file if specified
    if parsed_args.config_file:
        print(f"Loading configuration from: {parsed_args.config_file}")
        with open(parsed_args.config_file, 'r') as f:
            config_from_file = yaml.safe_load(f)
        
        # Update parsed_args with values from config file
        # Values explicitly passed via command line will override config file values
        # To do this, re-parse with config as defaults, or update namespace carefully
        arg_dict = vars(parsed_args)
        for key, value in config_from_file.items():
            # Only update if the command-line arg was not set (i.e., it's at its default)
            # This is a bit tricky because argparse doesn't easily tell us if an arg was explicitly set.
            # A common way is to set parser defaults to a unique sentinel value.
            # For simplicity here, config values will override defaults if not specified in CLI.
            # More robust: parser.set_defaults(**config_from_file); parsed_args = parser.parse_args()
            if key in arg_dict: # Check if it's a valid arg
                 # If CLI arg is different from default, it was likely set.
                 # This logic is not perfect for all types of defaults.
                 if arg_dict[key] == parser.get_default(key):
                    setattr(parsed_args, key, value)
    
    # Check for required arguments after config load
    missing_required = []
    for req_arg in parser.required_args_after_config:
        if getattr(parsed_args, req_arg) is None:
            missing_required.append(req_arg)
    if missing_required:
        parser.error(f"The following arguments are required: {', '.join(missing_required)}. "
                     "Please provide them via command line or config file.")

    main(parsed_args)
