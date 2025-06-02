import argparse
import torch
import torch.nn as nn # For nn.DataParallel
import pandas as pd
import numpy as np
from pathlib import Path
import json # For saving metrics
import random # For selecting CAM images
import cv2
import matplotlib.pyplot as plt
import os # For os.cpu_count()
import sys
from collections import OrderedDict # For loading state_dict

from torch.utils.data import DataLoader
import torchmetrics

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pneumonia_cam.utils import get_device, format_time 
from pneumonia_cam.data_loader import PneumoniaDataset, default_val_transforms 
from pneumonia_cam.model import PneumoniaModelCAM

def define_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Detection Model and Generate CAMs.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory (e.g., Processed/, containing dataset_stats.json and train/val splits).')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to the CSV file with labels (e.g., stage_2_train_labels.csv).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation metrics and CAM visualizations.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split to evaluate on.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for DataLoader. Defaults to os.cpu_count().')
    
    # Model parameters (defaults provided, but should ideally match checkpoint's saved args)
    parser.add_argument('--img_size', type=int, default=224, help='Image size used for training.')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels.')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='TIMM model architecture name.')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes.') # Should be 1 for binary
    parser.add_argument('--classifier_hidden_dim', type=int, default=512, help='Classifier hidden dimension.')
    parser.add_argument('--classifier_dropout_rate', type=float, default=0.3, help='Classifier dropout rate.')

    parser.add_argument('--num_cam_images', type=int, default=5, help='Number of random images for CAM generation. 0 to disable.')
    return parser

def visualize_and_save_cam(
    original_image_tensor: torch.Tensor, # Should be the tensor BEFORE normalization for best viz
    cam_map_np: np.ndarray,
    logits: torch.Tensor, # Raw logits for the image
    output_path: Path,
    threshold: float = 0.0 # Logits threshold for positive prediction
):
    img_display = original_image_tensor.cpu().squeeze() 
    if img_display.ndim == 3 and img_display.shape[0] == 1: 
        img_display = img_display.squeeze(0)
    elif img_display.ndim == 3 and img_display.shape[0] != 1: 
         img_display = img_display.permute(1,2,0)
    
    img_np = img_display.numpy().astype(np.float32)

    # Denormalize if mean/std were part of original_image_tensor's transforms
    # For simplicity, assuming original_image_tensor is [0,1] or can be displayed as is.
    # If it was normalized, we'd need mean/std to reverse it for accurate visualization.
    # The current PneumoniaDataset.__getitem__ returns normalized tensor.
    # For true original, one would need to load from .npy and just ToTensor().
    if img_np.min() < -1.0 or img_np.max() > 1.0 + 1e-5 : # Heuristic for normalized image
         # Attempt to shift towards [0,1] if it looks like it was normalized with mean~0.5, std~0.25
         if -2.0 < img_np.min() < -0.1 and img_np.max() < 2.0 : # very rough guess for imagenet-style norm
            img_np = (img_np * 0.229) + 0.485 # Example reverse for one common normalization
            img_np = np.clip(img_np, 0, 1)


    if img_np.min() < 0.0 or img_np.max() > 1.0: # If still not in range, clip after trying to denorm
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-8)
        img_np = np.clip(img_np, 0, 1)


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_np, cmap='bone')
    ax[0].set_title("Original Image (Processed)")
    ax[0].axis('off')

    ax[1].imshow(img_np, cmap='bone')
    ax[1].imshow(cam_map_np, alpha=0.5, cmap='jet')
    
    prob = torch.sigmoid(logits).item()
    is_positive = logits.item() > threshold
    ax[1].set_title(f"Prediction Positive: {is_positive} (Prob: {prob:.3f}, Logit: {logits.item():.3f})")
    ax[1].axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    # print(f"CAM saved to {output_path}")
    plt.close(fig)

def main(args):
    output_dir = Path(args.output_dir)
    cam_output_dir = output_dir / 'cam_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Try to load model architecture args from checkpoint, otherwise use command line args
    # This makes evaluation more robust if model details change.
    ckpt_args = checkpoint.get('args', {}) # Assuming 'args' (dict) was saved in checkpoint
    
    model_name = ckpt_args.get('model_name', args.model_name)
    in_channels = ckpt_args.get('in_channels', args.in_channels)
    img_size = ckpt_args.get('img_size', args.img_size)
    num_classes = ckpt_args.get('num_classes', args.num_classes)
    classifier_hidden_dim = ckpt_args.get('classifier_hidden_dim', args.classifier_hidden_dim)
    classifier_dropout_rate = ckpt_args.get('classifier_dropout_rate', args.classifier_dropout_rate)
    # pos_weight for criterion is not needed for model init here, but good to log if it was used.
    # criterion_pos_weight = ckpt_args.get('pos_weight', None) 

    print(f"Initializing model '{model_name}' for evaluation...")
    model = PneumoniaModelCAM(
        pretrained_model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        classifier_hidden_dim=classifier_hidden_dim,
        classifier_dropout_rate=classifier_dropout_rate
        # criterion_pos_weight is part of model's criterion, not needed for re-init if not training
    )

    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict.keys()):
        print("Adjusting state_dict from DataParallel model.")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Data
    print(f"Loading labels from: {args.label_csv} for split: {args.split}")
    if not Path(args.label_csv).exists():
        raise FileNotFoundError(f"Label CSV file not found: {args.label_csv}")
    labels_df = pd.read_csv(args.label_csv)

    if not labels_df.empty:
        if labels_df['patientId'].duplicated().any():
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
    else:
        raise ValueError("Label CSV file is empty.")

    print(f"Creating DataLoader for data in: {args.data_dir}, split: {args.split}")
    # PneumoniaDataset expects default_val_transforms to be applied.
    # default_val_transforms includes ToTensor and Normalize (internally by PneumoniaDataset now)
    eval_dataset = PneumoniaDataset(
        root_dir=Path(args.data_dir),
        split=args.split,
        dataframe=deduplicated_labels_df,
        transform=default_val_transforms # Pass the default val transforms
    )
    
    if len(eval_dataset) == 0:
        print(f"Warning: Evaluation dataset for split '{args.split}' is empty. Check data_dir and label_csv.")
        # Create dummy metrics and exit or proceed with CAM if possible
        metrics = {"accuracy": 0, "auroc": 0, "sensitivity": 0, "specificity": 0, 
                   "tn": 0, "fp": 0, "fn": 0, "tp": 0, "samples": 0}
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_dir / 'metrics.json'}")
        if args.num_cam_images > 0: print("Skipping CAM generation as dataset is empty.")
        return


    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Metrics
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    cm_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device) # num_classes=2 for TN,FP,FN,TP

    all_preds_proba = []
    all_labels_list = [] # Renamed to avoid conflict

    print(f"Starting evaluation on {args.split} split...")
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc=f"Evaluating {args.split}", unit="batch"):
            images, labels = images.to(device), labels.to(device).squeeze(-1).int() # Squeeze label for metrics
            
            logits, _ = model(images) # Logits are [B]
            probs = torch.sigmoid(logits)

            accuracy_metric.update(probs, labels)
            auroc_metric.update(probs, labels)
            cm_metric.update(probs, labels)
            
            all_preds_proba.extend(probs.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    acc = accuracy_metric.compute().item()
    auroc = auroc_metric.compute().item()
    cm = cm_metric.compute().cpu().numpy()

    if cm.size == 4: # Ensure cm is 2x2
        tn, fp, fn, tp = cm.ravel()
    else: # Handle cases with no samples in one class or empty dataset
        tn, fp, fn, tp = 0,0,0,0
        print(f"Warning: Confusion matrix is not 2x2 ({cm.shape}). Metrics might be skewed or zero.")


    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": acc, "auroc": auroc,
        "sensitivity_recall": sensitivity, "specificity": specificity,
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "total_samples": len(all_labels_list)
    }
    print(f"Evaluation Metrics ({args.split} split):")
    for k, v in metrics.items(): print(f"  {k}: {v}")

    with open(output_dir / f"metrics_{args.split}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_dir / f'metrics_{args.split}.json'}")

    # CAM Visualization
    if args.num_cam_images > 0 and len(eval_dataset) > 0:
        print(f"Generating CAMs for {args.num_cam_images} random images...")
        # Ensure we don't request more images than available
        num_to_visualize = min(args.num_cam_images, len(eval_dataset))
        random_indices = random.sample(range(len(eval_dataset)), num_to_visualize)

        for i, idx in enumerate(tqdm(random_indices, desc="Generating CAMs", unit="image")):
            img_tensor, label = eval_dataset[idx] # This tensor is already transformed by PneumoniaDataset (incl. normalization)
            
            # For visualize_and_save_cam, we want original_image_tensor to be less processed if possible.
            # However, PneumoniaDataset applies normalization. We can pass the normalized one for now,
            # or try to get an unnormalized version if needed (would require changing dataset or loading manually).
            # The current CAM method uses the model's device and dtype.
            
            cam_np, cam_logits = model.generate_cam(
                img_tensor.unsqueeze(0), # Add batch dim, model.generate_cam handles device
                target_size=(args.img_size, args.img_size)
            )

            if cam_np is not None and cam_logits is not None:
                actual_label = label.item() # label is a tensor like tensor([0.])
                # Use patient ID if possible for filename, otherwise index
                patient_id_for_cam = eval_dataset.samples[idx][0].stem # Get patient_id from Path object
                cam_filename = f'cam_{args.split}_sample_{patient_id_for_cam}_idx_{idx}_label_{actual_label:.0f}_pred_{torch.sigmoid(cam_logits).item():.2f}.png'
                visualize_and_save_cam(
                    img_tensor, # Pass the (normalized) tensor as returned by dataset
                    cam_np, 
                    cam_logits, 
                    cam_output_dir / cam_filename
                )
        print(f"CAM visualizations saved to {cam_output_dir}")
    else:
        print("Skipping CAM generation.")
    print("Evaluation finished.")

if __name__ == '__main__':
    parser = define_arg_parser()
    args = parser.parse_args()
    main(args)
