import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys

# The script is now part of the pneumonia_cam package (in pneumonia_cam.bin)
# Relative imports should work when run as part of the package.
# The sys.path manipulation is typically not needed if the package is installed
# or if run using `python -m pneumonia_cam.bin.preprocess_data`.

from ..data_loader import process_and_save_dicom_image
# from ..utils import is_directory_empty # Not strictly needed for this script's core logic

def main(args):
    dicom_dir = Path(args.dicom_dir)
    label_path = Path(args.label_path)
    output_dir = Path(args.output_dir)
    val_split_ratio = args.val_split_ratio
    img_resize_dim = (args.img_resize_dim, args.img_resize_dim) # Make it a tuple

    print(f"Starting preprocessing...")
    print(f"DICOM Directory: {dicom_dir}")
    print(f"Label Path: {label_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Validation Split Ratio: {val_split_ratio}")
    print(f"Image Resize Dimensions: {img_resize_dim}")

    # a. Load labels from CSV
    if not label_path.exists():
        print(f"Error: Label file not found at {label_path}")
        return
    labels_df = pd.read_csv(label_path)
    print(f"Loaded labels: {len(labels_df)} rows.")

    # b. Perform label deduplication
    if not labels_df.empty:
        # Ensure 'patientId' and 'Target' exist; handle potential missing x, y, width, height for Target=0
        if 'x' not in labels_df.columns: labels_df['x'] = np.nan # Add if missing
        
        # Create a nonnull_count based on relevant columns for deduplication
        # If Target is 0, x,y,width,height might be NaN. These are relevant for Target=1.
        # For deduplication, we mostly care about having one definitive row per patientId.
        # The original script's logic for nonnull_count might be simplified if we only
        # care about 'Target' for stratification and 'patientId' for uniqueness.
        # Let's stick to the original logic for now.
        relevant_cols_for_dedup = ['x', 'y', 'width', 'height', 'Target']
        for col in relevant_cols_for_dedup: # Ensure all relevant columns exist
            if col not in labels_df.columns:
                labels_df[col] = np.nan if col != 'Target' else 0 # Sensible defaults
        
        labels_df["nonnull_count"] = labels_df[relevant_cols_for_dedup].notnull().sum(axis=1)
        
        # Keep the row with the most non-null values for each patientId.
        # If counts are equal, idxmax() keeps the first occurrence.
        deduplicated_labels_df = labels_df.loc[labels_df.groupby("patientId")["nonnull_count"].idxmax()]
        deduplicated_labels_df = deduplicated_labels_df.drop(columns="nonnull_count")
        deduplicated_labels_df.reset_index(drop=True, inplace=True)
        print(f"Deduplicated labels: {len(deduplicated_labels_df)} unique patient entries.")
    else:
        print("Labels DataFrame is empty. Cannot proceed.")
        return

    # c. Split patient IDs into training and validation sets
    patient_ids = deduplicated_labels_df["patientId"].unique()
    # Stratify by 'Target' to ensure similar class distribution in train/val
    # Need to map patient_ids to their targets for stratification
    patient_id_to_target_map = deduplicated_labels_df.set_index('patientId')['Target'].to_dict()
    targets_for_split = [patient_id_to_target_map[pid] for pid in patient_ids]

    train_patient_ids, val_patient_ids = train_test_split(
        patient_ids,
        test_size=val_split_ratio,
        random_state=42, # For reproducibility
        stratify=targets_for_split
    )
    print(f"Train patient IDs: {len(train_patient_ids)}, Validation patient IDs: {len(val_patient_ids)}")

    # d. Initialize accumulators for mean/std calculation (from training set only)
    total_sum_of_pixels = 0.0
    total_sum_of_squared_pixels = 0.0
    training_pixel_count = 0

    # e. Iterate through all patient IDs
    print("Processing and saving images...")
    for patient_id in tqdm(deduplicated_labels_df["patientId"], desc="Processing DICOMs"):
        patient_row = deduplicated_labels_df[deduplicated_labels_df["patientId"] == patient_id].iloc[0]
        label = int(patient_row["Target"])
        
        split_set = "train" if patient_id in train_patient_ids else "val"
        
        dicom_file_path = dicom_dir / f"{patient_id}.dcm"
        output_npy_path = output_dir / split_set / str(label) / f"{patient_id}.npy"

        if not dicom_file_path.exists():
            print(f"Warning: DICOM file {dicom_file_path} not found for patient {patient_id}. Skipping.")
            continue

        processed_array = process_and_save_dicom_image(
            dicom_path=dicom_file_path,
            output_npy_path=output_npy_path,
            resize_dim=img_resize_dim
        )

        if processed_array is not None and split_set == "train":
            # Ensure array is float64 for precise sum accumulation
            processed_array_float64 = processed_array.astype(np.float64)
            total_sum_of_pixels += np.sum(processed_array_float64)
            total_sum_of_squared_pixels += np.sum(np.square(processed_array_float64))
            training_pixel_count += processed_array_float64.size
            
    # f. Calculate global mean and standard deviation from training set
    if training_pixel_count == 0:
        print("Error: No pixels processed from the training set. Cannot calculate mean/std.")
        mean_val = 0.0
        std_val = 0.0
    else:
        mean_val = total_sum_of_pixels / training_pixel_count
        # Variance = E[X^2] - (E[X])^2
        variance = (total_sum_of_squared_pixels / training_pixel_count) - (mean_val ** 2)
        if variance < 0: # Due to floating point inaccuracies, variance can be slightly negative
            print(f"Warning: Calculated variance is negative ({variance:.8f}). Clamping to 0.")
            variance = 0.0
        std_val = np.sqrt(variance)
    
    print(f"Calculated from training set: Mean = {mean_val:.6f}, Std = {std_val:.6f}, Pixels = {training_pixel_count}")

    # g. Save mean and std to dataset_stats.json
    stats = {"mean": mean_val, "std": std_val, "training_pixel_count": training_pixel_count}
    stats_path = output_dir / "dataset_stats.json"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output_dir itself exists
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Dataset statistics saved to {stats_path}")
    print("Preprocessing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DICOM images for pneumonia detection.")
    parser.add_argument("--dicom_dir", type=str, required=True, help="Path to the directory with raw DICOM images.")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the CSV file with labels (e.g., stage_2_train_labels.csv).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed .npy files and dataset_stats.json.")
    parser.add_argument("--val_split_ratio", type=float, default=0.2, help="Fraction of data to use for the validation set.")
    parser.add_argument("--img_resize_dim", type=int, default=256, help="Target dimension for image resizing (e.g., 256 for 256x256).")
    
    # Example usage from pneumonia_cam_project directory:
    # python scripts/preprocess_data.py --dicom_dir Pneumonia/stage_2_train_images --label_path Pneumonia/stage_2_train_labels.csv --output_dir Processed --val_split_ratio 0.2 --img_resize_dim 256

    parsed_args = parser.parse_args()
    main(parsed_args)
