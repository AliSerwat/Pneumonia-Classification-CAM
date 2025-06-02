import argparse
from pathlib import Path
import time
import random
from typing import Union, Tuple, Optional, List, Dict, Callable # Added Union

import torch
import torch.nn as nn # Added import for nn
import torch.onnx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom # For visualize_dicom_images, if included

from torch.utils.data import DataLoader
import torchmetrics
from sklearn.metrics import confusion_matrix, roc_curve, auc as sklearn_auc # For more detailed metrics
from tqdm.auto import tqdm # Added tqdm for progress bars

# Assuming model.py, data_loader.py, utils.py are in the same package
from .model import PneumoniaModelCAM # Or your specific model class
from .data_loader import PneumoniaDataset, get_val_transforms, DEFAULT_MEAN, DEFAULT_STD
from .utils import get_device, format_time, DEVICE

# ============================ Model Loading ============================ #

def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_class: torch.nn.Module = PneumoniaModelCAM, # Allow specifying model class
    device: torch.device = DEVICE,
    **model_kwargs
) -> torch.nn.Module:
    """
    Loads a model from a saved checkpoint.
    The checkpoint is expected to contain 'model_state_dict' and 'config'.

    Args:
        checkpoint_path (Union[str, Path]): Path to the .pth checkpoint file.
        model_class (torch.nn.Module): The class of the model to instantiate.
        device (torch.device): Device to load the model onto.
        **model_kwargs: Additional keyword arguments to pass to the model constructor
                        if 'config' is not found in checkpoint or to override.

    Returns:
        torch.nn.Module: The loaded model.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration from checkpoint if available
    cfg_from_checkpoint = checkpoint.get('config')

    model_args = {} # Initialize model_args
    if cfg_from_checkpoint:
        print("Found model configuration in checkpoint.")
        # Ensure essential args are present, allow overrides from model_kwargs
        model_args = {
            'pretrained_model_name': cfg_from_checkpoint.model_name,
            'num_classes': 1, # Assuming binary for this project from context
            'in_channels': cfg_from_checkpoint.in_channels,
            'img_size': cfg_from_checkpoint.img_size,
            'classifier_hidden_dim': cfg_from_checkpoint.classifier_hidden_dim,
            'classifier_dropout_rate': cfg_from_checkpoint.classifier_dropout_rate,
        }
        model_args.update(model_kwargs) # Override with any explicit kwargs
        model = model_class(**model_args)
    else:
        print("WARNING: Model configuration not found in checkpoint. Using provided model_kwargs or defaults.")
        if not model_kwargs:
            # Try to infer from a potentially simpler model state dict or raise error
            print("ERROR: No config in checkpoint and no model_kwargs provided. Cannot instantiate model.")
            # A common practice is to save enough model args in config to reinstantiate
            # For now, if this happens, it's an issue with the checkpoint or call.
            raise ValueError("Cannot determine model architecture. Checkpoint needs 'config' or provide model_kwargs.")
        model_args.update(model_kwargs) # Ensure model_kwargs are stored in model_args for printout
        model = model_class(**model_args)

    # Handle DataParallel state_dict
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("Adjusting DataParallel state_dict keys.")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model '{model_args.get('pretrained_model_name', 'N/A')}' loaded successfully and set to eval mode.")
    return model

# ============================ CAM Generation ============================ #

def generate_cam(
    model: PneumoniaModelCAM,
    img_tensor: torch.Tensor, # Expected single image tensor [C, H, W] or [H,W]
    target_size: Optional[Tuple[int, int]] = None # (width, height) for resizing CAM
) -> Tuple[np.ndarray, float]:
    """
    Generates a Class Activation Map (CAM) for a given image tensor using the model.
    This version uses the weights of the final linear layer of the classifier.

    Args:
        model (PneumoniaModelCAM): The trained model instance.
        img_tensor (torch.Tensor): Input image tensor (e.g., [C, H, W] or [H,W] for grayscale).
        target_size (Optional[Tuple[int, int]]): The target (width, height) to resize the CAM to.
                                                 If None, uses original image size if possible or feature map size.

    Returns:
        Tuple[np.ndarray, float]:
            - cam_normalized (np.ndarray): The normalized CAM as a 2D NumPy array.
            - probability (float): The predicted probability for the positive class.
    """
    model.eval()
    model_device = next(model.parameters()).device # Get device model is on

    # Prepare image tensor
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.from_numpy(img_tensor).float()

    processed_img_tensor = img_tensor.to(model_device)
    # Check if model is in half precision (common attribute name might be model.dtype)
    # This check is a bit heuristic; a more robust way might involve checking a parameter's dtype.
    if hasattr(model, 'dtype') and model.dtype == torch.float16:
        processed_img_tensor = processed_img_tensor.half()
    elif next(model.parameters()).dtype == torch.float16: # Check parameter dtype
        processed_img_tensor = processed_img_tensor.half()


    if processed_img_tensor.ndim == 2: # Grayscale [H, W] -> [1, H, W]
        processed_img_tensor = processed_img_tensor.unsqueeze(0)
    if processed_img_tensor.ndim != 3:
        raise ValueError(f"Input img_tensor must be 2D [H,W] or 3D [C,H,W], got {processed_img_tensor.ndim}D")

    input_batch = processed_img_tensor.unsqueeze(0) # [C, H, W] -> [1, C, H, W]

    # Get feature maps and logits
    with torch.no_grad():
        logits, feature_maps = model(input_batch) # feature_maps: [1, C_feat, H_feat, W_feat]

    # Get weights from the final linear layer of the classifier
    # Accessing the EnhancedClassifier's feature_map_processor, then its last layer
    final_fc_layer = model.classifier.feature_map_processor[-1]
    if not isinstance(final_fc_layer, nn.Linear): # Use nn after import
        raise TypeError("The last layer of EnhancedClassifier's feature_map_processor is not nn.Linear.")

    classifier_weights = final_fc_layer.weight.squeeze()
    if classifier_weights.ndim == 0:
        classifier_weights = classifier_weights.unsqueeze(0)


    squeezed_feature_maps = feature_maps.squeeze(0)

    if classifier_weights.ndim > 1 and classifier_weights.shape[0] > 1:
        pred_class_idx = torch.argmax(logits, dim=1).item() if logits.ndim > 1 and logits.shape[1] > 1 else 0
        weights_for_cam = classifier_weights[pred_class_idx, :]
    else:
        weights_for_cam = classifier_weights

    cam = torch.einsum('c,chw->hw', weights_for_cam, squeezed_feature_maps)
    cam_np = cam.cpu().numpy()

    cam_np = cam_np - np.min(cam_np)
    if np.max(cam_np) > 1e-6:
        cam_np = cam_np / np.max(cam_np)
    else:
        cam_np = np.zeros_like(cam_np)

    if target_size:
        cam_resized = cv2.resize(cam_np, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        if img_tensor.ndim >= 2:
            original_h = img_tensor.shape[-2]
            original_w = img_tensor.shape[-1]
            cam_resized = cv2.resize(cam_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_resized = cam_np

    probability = torch.sigmoid(logits.squeeze()).item()
    return cam_resized, probability

# ============================ Visualization ============================ #

def visualize_cam_overlay(
    original_image_np: np.ndarray, # Expected (H, W) or (H, W, C)
    cam_np: np.ndarray, # Expected (H, W)
    prediction_prob: float,
    actual_label: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 5)
):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    img_display = original_image_np.astype(np.float32)
    if img_display.min() < 0 or img_display.max() > 1:
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-6)

    if img_display.ndim == 3 and img_display.shape[-1] == 1:
        img_display = img_display.squeeze(-1)

    axes[0].imshow(img_display, cmap='bone')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(img_display, cmap='bone')
    axes[1].imshow(cam_np, cmap='jet', alpha=0.5)

    title_str = f"Prediction: {prediction_prob:.2f} ('Pneumonia' if >0.5)"
    if actual_label is not None:
        title_str += f"\nActual: {'Pneumonia' if actual_label == 1 else 'Normal'}"
    axes[1].set_title(title_str)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show(block=False) # Use block=False to avoid blocking in scripts
    plt.pause(0.1) # Pause to allow plot to render

# ============================ Evaluation Metrics ============================ #

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = DEVICE
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    all_preds_proba = []
    all_labels = []

    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    auroc_metric = torchmetrics.AUROC(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    specificity_metric = torchmetrics.Specificity(task="binary").to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

            logits, _ = model(imgs)
            if logits.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 1:
                logits = logits.unsqueeze(1)

            preds_proba = torch.sigmoid(logits)

            all_preds_proba.append(preds_proba.cpu())
            all_labels.append(labels.cpu().int())

            accuracy_metric.update(preds_proba, labels.int())
            auroc_metric.update(preds_proba, labels.int())
            precision_metric.update(preds_proba, labels.int())
            recall_metric.update(preds_proba, labels.int())
            specificity_metric.update(preds_proba, labels.int())
            f1_metric.update(preds_proba, labels.int())

    metrics = {
        "accuracy": accuracy_metric.compute().item(),
        "auroc": auroc_metric.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall (sensitivity)": recall_metric.compute().item(),
        "specificity": specificity_metric.compute().item(),
        "f1_score": f1_metric.compute().item(),
    }

    if all_preds_proba and all_labels:
        y_true = torch.cat(all_labels).numpy().flatten()
        y_pred_proba = torch.cat(all_preds_proba).numpy().flatten()
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred_binary)
        metrics["confusion_matrix (tn, fp, fn, tp)"] = cm.ravel().tolist()
        print("Confusion Matrix (tn, fp, fn, tp):", metrics["confusion_matrix (tn, fp, fn, tp)"])

    return metrics

# ============================ ONNX Export ============================ #
def export_to_onnx(
    model: torch.nn.Module,
    dummy_input_shape: Tuple[int, int, int, int],
    onnx_path: Union[str, Path],
    opset_version: int = 12
):
    model.eval()
    model.to(torch.device('cpu'))

    dummy_input = torch.randn(*dummy_input_shape, device='cpu')
    output_names = ["logits", "feature_maps"] if isinstance(model, PneumoniaModelCAM) else ["output"]

    dynamic_axes_config = {'input_image': {0: 'batch_size'}, output_names[0]: {0: 'batch_size'}}
    if len(output_names) > 1: # Add dynamic axis for the second output if it exists
        dynamic_axes_config[output_names[1]] = {0: 'batch_size'}


    try:
        print(f"Exporting model to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=output_names,
            dynamic_axes=dynamic_axes_config
        )
        print(f"✅ Model exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")

# ============================ Main Evaluation Script ============================ #

def main_evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Detection Model and Generate CAMs")

    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument('--project_root_dir', type=str, default='.', help="Root directory of the project.")
    parser.add_argument('--dataset_csv_path', type=str, default='Pneumonia/stage_2_train_labels.csv', help="Path to dataset CSV for evaluation.")
    parser.add_argument('--processed_data_dir', type=str, default='Processed', help="Directory with processed .npy files for evaluation.")
    parser.add_argument('--eval_split', type=str, default='val', choices=['train', 'val', 'test'], help="Dataset split to evaluate ('val' or 'test' typically).")

    parser.add_argument('--model_name', type=str, default=None, help="Name of TIMM model (if not in checkpoint or to override).")
    parser.add_argument('--img_size', type=int, default=None, help="Image size (if not in checkpoint or to override).")
    parser.add_argument('--in_channels', type=int, default=1, help="Input channels (if not in checkpoint or to override).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2 if os.cpu_count() is not None and os.cpu_count() > 1 else 1, help="DataLoader workers.")

    parser.add_argument('--generate_cam', action='store_true', help="Generate and visualize CAM for random samples.")
    parser.add_argument('--num_cam_samples', type=int, default=5, help="Number of random samples for CAM visualization.")

    parser.add_argument('--export_onnx', action='store_true', help="Export the loaded model to ONNX format.")
    parser.add_argument('--onnx_filename', type=str, default="pneumonia_model.onnx", help="Filename for the exported ONNX model.")

    args = parser.parse_args()

    model_kwargs_from_args = {}
    if args.model_name: model_kwargs_from_args['pretrained_model_name'] = args.model_name
    if args.img_size: model_kwargs_from_args['img_size'] = args.img_size
    if args.in_channels: model_kwargs_from_args['in_channels'] = args.in_channels

    model = load_model_from_checkpoint(args.checkpoint_path, device=DEVICE, **model_kwargs_from_args)

    eval_data_path = Path(args.project_root_dir) / args.processed_data_dir / args.eval_split
    if not eval_data_path.exists():
        raise FileNotFoundError(f"Evaluation data split directory not found: {eval_data_path}")

    labels_df_path = Path(args.project_root_dir) / args.dataset_csv_path
    if not labels_df_path.exists():
        raise FileNotFoundError(f"Labels CSV for evaluation not found: {labels_df_path}")
    labels_df = pd.read_csv(labels_df_path)
    labels_df = labels_df.assign(
        nonnull_count=labels_df.drop(columns="patientId", errors='ignore').notnull().sum(axis=1)
    ).sort_values(
        by=["patientId", "nonnull_count"], ascending=[True, False]
    ).drop_duplicates(
        subset="patientId", keep="first"
    ).drop(columns="nonnull_count")

    transform_img_size = model.img_size if hasattr(model, 'img_size') else args.img_size
    if transform_img_size is None:
        transform_img_size = DEFAULT_IMG_SIZE
        print(f"Warning: Using default img_size {transform_img_size} for transforms.")

    eval_transforms = get_val_transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)

    eval_dataset = PneumoniaDataset(
        dataframe=labels_df,
        processed_root_dir=eval_data_path,
        transform=eval_transforms,
        dataset_split=args.eval_split
    )
    if len(eval_dataset) == 0:
        print(f"Warning: Evaluation dataset for split '{args.eval_split}' is empty. Skipping evaluation.")
    else:
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        print(f"Evaluating on {len(eval_dataset)} samples from '{args.eval_split}' split...")
        metrics = evaluate_model(model, eval_loader, device=DEVICE)
        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            if isinstance(v, (list, np.ndarray)): # Check if list or numpy array for direct print
                 print(f"{k}: {v}")
            else: # Assume float for formatting
                 print(f"{k}: {v:.4f}")

    if args.generate_cam and len(eval_dataset) > 0:
        print(f"\n--- Generating CAM for {args.num_cam_samples} random samples ---")
        # Ensure eval_dataset.samples is populated
        if not eval_dataset.samples:
            print("Warning: eval_dataset.samples is empty. Cannot generate CAMs.")
            return # or handle error appropriately

        random_indices = random.sample(range(len(eval_dataset)), min(args.num_cam_samples, len(eval_dataset)))

        for i, idx in enumerate(random_indices):
            img_tensor, label_tensor = eval_dataset[idx]

            original_npy_path = eval_dataset.samples[idx][0]
            original_img_np = PneumoniaDataset._load_file(original_npy_path)

            if original_img_np.ndim == 3 and original_img_np.shape[0] == 1:
                original_img_np = original_img_np.squeeze(0)

            print(f"Sample {i+1}/{args.num_cam_samples} (Index: {idx}, Path: {original_npy_path})")

            cam_output, probability = generate_cam(model, img_tensor, target_size=(original_img_np.shape[1], original_img_np.shape[0]))
            visualize_cam_overlay(original_img_np, cam_output, probability, actual_label=int(label_tensor.item()))
            # time.sleep(0.5) # Removed for non-blocking behavior by default

    if args.export_onnx:
        onnx_batch, onnx_channels, onnx_height, onnx_width = 1, DEFAULT_IN_CHANNELS, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE # Fallbacks
        if hasattr(model, 'in_channels'): onnx_channels = model.in_channels
        elif args.in_channels: onnx_channels = args.in_channels
        if hasattr(model, 'img_size'): onnx_height = onnx_width = model.img_size
        elif args.img_size: onnx_height = onnx_width = args.img_size

        dummy_shape = (onnx_batch, onnx_channels, onnx_height, onnx_width)
        onnx_file_path = Path(args.project_root_dir) / args.onnx_filename
        export_to_onnx(model, dummy_shape, onnx_file_path)

    print("\nEvaluation script finished.")

if __name__ == "__main__":
    # Define fallbacks for CLI examples if these constants are not globally defined
    DEFAULT_IN_CHANNELS = 1
    DEFAULT_IMG_SIZE = 256

    # Example commands (run from project root e.g. pneumonia_cam_project/):
    #
    # To evaluate a model on the 'val' split:
    # python -m pneumonia_cam.evaluate --checkpoint_path experiments/your_model_run/best_model_epoch_X.pth --processed_data_dir Processed --eval_split val
    #
    # To also generate CAMs:
    # python -m pneumonia_cam.evaluate --checkpoint_path path/to/best_model.pth --generate_cam --num_cam_samples 3
    #
    # To also export to ONNX:
    # python -m pneumonia_cam.evaluate --checkpoint_path path/to/best_model.pth --export_onnx --onnx_filename my_pneumonia_model.onnx
    main_evaluate()
