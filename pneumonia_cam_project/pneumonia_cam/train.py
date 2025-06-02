import argparse
import os
import time
from pathlib import Path
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchinfo import summary # For model summary, if desired
import pandas as pd # Added missing import for pandas
from tqdm.auto import tqdm # Ensure tqdm is imported

# Assuming model.py, data_loader.py, utils.py are in the same package (pneumonia_cam)
from .model import PneumoniaModelCAM
from .data_loader import (
    PneumoniaDataset,
    get_train_transforms,
    get_val_transforms,
    create_weighted_sampler,
    normalize_and_save_images, # If preprocessing is part of the train script
    DEFAULT_MEAN, DEFAULT_STD
)
from .utils import get_device, seed_everything, format_time, DEVICE

# Default configuration values (can be overridden by argparse)
DEFAULT_IMG_SIZE = 256
DEFAULT_BATCH_SIZE = 32 # Reduced from 64 for wider compatibility
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_MODEL_NAME = "efficientnet_b0" # A reasonably small default

def train_model(config):
    """
    Main training function for the Pneumonia Detection Model.

    Args:
        config (argparse.Namespace): Configuration object with all training parameters.
    """
    seed_everything(config.seed)

    # Paths
    project_root = Path(config.project_root_dir)
    dataset_csv_path = project_root / config.dataset_csv_path
    raw_dicom_dir = project_root / config.raw_dicom_dir

    processed_data_dir = project_root / config.processed_data_dir
    save_path = project_root / config.save_dir / f"{config.model_name}_run_{time.strftime('%Y%m%d_%H%M%S')}"
    save_path.mkdir(parents=True, exist_ok=True)

    logs_path = save_path / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(logs_path))

    print(f"🚀 Starting training run. Configuration: {config}")
    print(f"💾 Checkpoints and logs will be saved to: {save_path}")

    # --- 1. Data Preprocessing (Optional, if enabled) ---
    if config.preprocess_data:
        print(f"🔄 Starting data preprocessing from '{raw_dicom_dir}' to '{processed_data_dir}'...")
        if not dataset_csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at: {dataset_csv_path}")
        if not raw_dicom_dir.exists():
            raise FileNotFoundError(f"Raw DICOM directory not found at: {raw_dicom_dir}")

        labels_df = pd.read_csv(dataset_csv_path)
        # Ensure 'patientId' and 'Target' columns are present and correctly named as expected by normalize_and_save_images
        # Add any necessary preprocessing for labels_df here (e.g., deduplication from notebook)

        # Simplified deduplication from notebook:
        labels_df = labels_df.assign(
            nonnull_count=labels_df.drop(columns="patientId", errors='ignore').notnull().sum(axis=1)
        ).sort_values(
            by=["patientId", "nonnull_count"], ascending=[True, False]
        ).drop_duplicates(
            subset="patientId", keep="first"
        ).drop(columns="nonnull_count")

        calculated_mean, calculated_std = normalize_and_save_images(
            root_dicom_path=raw_dicom_dir,
            save_processed_path=processed_data_dir,
            labels_df=labels_df,
            target_img_size=config.img_size,
            train_split_ratio=0.8 # Hardcoded for now, could be a config
        )
        # Use calculated mean/std for transforms if they are valid, else use defaults
        mean_for_transforms = calculated_mean if calculated_mean != 0 else DEFAULT_MEAN
        std_for_transforms = calculated_std if calculated_std != 0 else DEFAULT_STD
        print(f"Preprocessing complete. Using mean={mean_for_transforms}, std={std_for_transforms} for DataLoaders.")
    else:
        print("⏩ Skipping data preprocessing. Assuming data is already processed in `processed_data_dir`.")
        mean_for_transforms = config.custom_mean if config.custom_mean is not None else DEFAULT_MEAN
        std_for_transforms = config.custom_std if config.custom_std is not None else DEFAULT_STD
        print(f"Using mean={mean_for_transforms}, std={std_for_transforms} for DataLoaders.")

    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found at: {processed_data_dir}. "
                                "Run with --preprocess_data or ensure data is preprocessed.")

    # --- 2. Data Loaders ---
    print("💾 Setting up DataLoaders...")
    # Full labels_df for mapping, PneumoniaDataset filters by structure
    if not dataset_csv_path.exists():
         raise FileNotFoundError(f"Dataset CSV for DataLoaders not found at: {dataset_csv_path}")
    labels_df_for_dataloaders = pd.read_csv(dataset_csv_path)
    # Apply same deduplication as in preprocessing to ensure consistency
    labels_df_for_dataloaders = labels_df_for_dataloaders.assign(
        nonnull_count=labels_df_for_dataloaders.drop(columns="patientId", errors='ignore').notnull().sum(axis=1)
    ).sort_values(
        by=["patientId", "nonnull_count"], ascending=[True, False]
    ).drop_duplicates(
        subset="patientId", keep="first"
    ).drop(columns="nonnull_count")

    train_transforms = get_train_transforms(config.img_size, mean_for_transforms, std_for_transforms)
    val_transforms = get_val_transforms(mean_for_transforms, std_for_transforms)

    # PneumoniaDataset expects processed_root_dir to be like .../Processed/train
    train_dataset_path = processed_data_dir / "train"
    val_dataset_path = processed_data_dir / "val"

    if not train_dataset_path.exists() or not val_dataset_path.exists():
        raise FileNotFoundError(f"Train ({train_dataset_path}) or Val ({val_dataset_path}) "
                                "subdirectories not found in processed_data_dir. Ensure preprocessing was successful.")

    train_dataset = PneumoniaDataset(
        dataframe=labels_df_for_dataloaders,
        processed_root_dir=train_dataset_path,
        transform=train_transforms,
        dataset_split="train"
    )
    val_dataset = PneumoniaDataset(
        dataframe=labels_df_for_dataloaders,
        processed_root_dir=val_dataset_path,
        transform=val_transforms,
        dataset_split="val"
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        raise ValueError("One of the datasets is empty. Check data paths and preprocessing.")

    train_sampler = create_weighted_sampler(train_dataset) if config.use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None, # Shuffle only if not using sampler
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True # Good for consistent batch sizes, esp. with multi-GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    print(f"Found {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- 3. Model, Optimizer, Criterion, Metrics ---
    print(f"🧠 Initializing model: {config.model_name}")
    model = PneumoniaModelCAM(
        pretrained_model_name=config.model_name,
        num_classes=1, # Binary classification
        in_channels=config.in_channels,
        img_size=config.img_size,
        classifier_hidden_dim=config.classifier_hidden_dim,
        classifier_dropout_rate=config.classifier_dropout_rate
    )

    # Model summary (optional)
    try:
        summary_str = summary(
            model,
            input_size=(config.batch_size, config.in_channels, config.img_size, config.img_size),
            verbose=0, # 0 for string output, 1 for print
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        )
        print(summary_str)
        with open(save_path / "model_summary.txt", "w") as f:
            f.write(str(summary_str))
    except Exception as e:
        print(f"⚠️ Could not print model summary: {e}")

    # DataParallel for multi-GPU if available and enabled
    if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1 and config.use_dataparallel:
        print(f"🚀 Using nn.DataParallel for {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Get criterion from model instance (handles pos_weight internally if defined)
    # If model is DataParallel, access original model via .module
    criterion_model_attr = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(criterion_model_attr, 'criterion') and criterion_model_attr.criterion is not None:
         criterion = criterion_model_attr.criterion.to(DEVICE)
         print("Using criterion from model attribute.")
    else: # Fallback if not defined on model (e.g. if PneumoniaModelCAM doesn't set self.criterion)
        print("Criterion not found on model, using default BCEWithLogitsLoss with pos_weight=1.0.")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=DEVICE))


    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Metrics
    train_accuracy = torchmetrics.Accuracy(task="binary").to(DEVICE)
    train_auroc = torchmetrics.AUROC(task="binary").to(DEVICE)
    val_accuracy = torchmetrics.Accuracy(task="binary").to(DEVICE)
    val_auroc = torchmetrics.AUROC(task="binary").to(DEVICE)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda' and config.use_amp))

    # --- 4. Training Loop ---
    best_val_metric = 0.0 # Using AUC as the primary metric for saving best model
    epochs_no_improve = 0

    overall_start_time = time.time()
    print(f"\n🚀 Starting training for {config.epochs} epochs on {DEVICE}.")

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_accuracy.reset()
        train_auroc.reset()

        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Train]", leave=False)
        for batch_idx, (imgs, labels) in train_progress_bar:
            imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE) # Ensure labels are [B,1]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda' and config.use_amp)):
                # PneumoniaModelCAM returns (logits, feature_maps)
                logits, _ = model(imgs)
                # Ensure logits shape matches labels for criterion
                if logits.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 1:
                    logits = logits.unsqueeze(1) # [B] -> [B,1]

                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * imgs.size(0)
            preds_proba = torch.sigmoid(logits.detach())
            train_accuracy.update(preds_proba, labels.int())
            train_auroc.update(preds_proba, labels.int())
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = train_accuracy.compute()
        epoch_train_auc = train_auroc.compute()

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_train_acc, epoch)
        writer.add_scalar("AUROC/train_epoch", epoch_train_auc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_accuracy.reset()
        val_auroc.reset()
        val_progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Val  ]", leave=False)
        with torch.no_grad():
            for imgs, labels in val_progress_bar:
                imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda' and config.use_amp)):
                    logits, _ = model(imgs)
                    if logits.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 1:
                        logits = logits.unsqueeze(1)
                    loss = criterion(logits, labels)

                running_val_loss += loss.item() * imgs.size(0)
                preds_proba = torch.sigmoid(logits.detach())
                val_accuracy.update(preds_proba, labels.int())
                val_auroc.update(preds_proba, labels.int())
                val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute()
        epoch_val_auc = val_auroc.compute()

        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val_epoch", epoch_val_acc, epoch)
        writer.add_scalar("AUROC/val_epoch", epoch_val_auc, epoch)

        epoch_duration_seconds = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{config.epochs} | Duration: {format_time(epoch_duration_seconds)} | "
            f"Train Loss: {avg_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | AUC: {epoch_train_auc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | AUC: {epoch_val_auc:.4f}"
        )

        # Checkpoint saving
        current_val_metric = epoch_val_auc # Using AUC for checkpointing
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_metric': best_val_metric,
                'config': config
            }
            model_save_name = save_path / f"best_model_epoch_{epoch+1}_auc_{best_val_metric:.4f}.pth"
            torch.save(checkpoint, model_save_name)
            print(f"🏅 Checkpoint saved: {model_save_name} (Val AUC: {best_val_metric:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"✋ Early stopping triggered at epoch {epoch+1} after {config.early_stopping_patience} epochs with no improvement.")
            break

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_val_metric) # Step with AUC for ReduceLROnPlateau
            # else: scheduler.step() # For other epoch-level schedulers

    writer.close()
    total_training_duration_seconds = time.time() - overall_start_time
    print("\n--- 🏆 Training Finished ---")
    print(f"Total actual training time: {format_time(total_training_duration_seconds)}")
    print(f"Best Validation Metric (AUC) achieved: {best_val_metric:.4f}")
    print(f"Checkpoints and logs saved in: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model")

    # Paths
    parser.add_argument('--project_root_dir', type=str, default='.', help="Root directory of the project.")
    parser.add_argument('--dataset_csv_path', type=str, default='Pneumonia/stage_2_train_labels.csv', help="Path to the dataset CSV file, relative to project_root_dir.")
    parser.add_argument('--raw_dicom_dir', type=str, default='Pneumonia/stage_2_train_images', help="Directory with raw DICOM files, relative to project_root_dir.")
    parser.add_argument('--processed_data_dir', type=str, default='Processed', help="Directory for processed .npy files, relative to project_root_dir.")
    parser.add_argument('--save_dir', type=str, default='experiments', help="Directory to save training runs/checkpoints, relative to project_root_dir.")

    # Preprocessing
    parser.add_argument('--preprocess_data', action='store_true', help="If set, run preprocessing. Otherwise, expects preprocessed data.")
    parser.add_argument('--custom_mean', type=float, default=None, help="Custom mean for normalization if not preprocessing.")
    parser.add_argument('--custom_std', type=float, default=None, help="Custom std for normalization if not preprocessing.")

    # Model
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help="Name of the TIMM pretrained model.")
    parser.add_argument('--in_channels', type=int, default=1, help="Number of input image channels (1 for grayscale, 3 for RGB).")
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help="Image size (height and width).")
    parser.add_argument('--classifier_hidden_dim', type=int, default=512, help="Hidden dimension for the classifier.")
    parser.add_argument('--classifier_dropout_rate', type=float, default=0.3, help="Dropout rate for the classifier.")

    # Training
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training and validation.")
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for AdamW optimizer.")
    parser.add_argument('--use_weighted_sampler', action='store_true', help="Use weighted random sampler for imbalanced data.")
    parser.add_argument('--use_scheduler', action='store_true', help="Use ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument('--use_amp', action='store_true', help="Use Automatic Mixed Precision (AMP) for training if CUDA is available.")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping.")

    # System
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2 if os.cpu_count() is not None and os.cpu_count() > 1 else 1, help="Number of workers for DataLoader.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--use_dataparallel', action='store_true', help="Use nn.DataParallel if multiple GPUs are available.")


    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    # Example command to run from project root (e.g., pneumonia_cam_project/):
    # python -m pneumonia_cam.train --preprocess_data --model_name efficientnet_b0 --epochs 2 --batch_size 16 --img_size 224 --save_dir training_runs
    #
    # To run without preprocessing (assuming 'Processed' dir exists):
    # python -m pneumonia_cam.train --model_name efficientnet_b0 --epochs 10 --batch_size 16 --img_size 224 --save_dir training_runs --processed_data_dir Processed
    main()
