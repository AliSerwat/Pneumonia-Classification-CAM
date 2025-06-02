import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader # Added DataLoader
from torchvision import transforms as t
from pathlib import Path
import numpy as np
import pandas as pd # For _create_patient_id_map
from typing import Optional, Callable, Tuple, List, Union, Dict
from collections import Counter # For create_weighted_sampler
import pydicom # For DICOM processing
import cv2 # For image resizing
import json # For loading dataset_stats.json

# Updated Transforms (Normalization removed, will be handled in Dataset)
# These are the default transforms if none are provided to create_dataloaders
default_train_transforms = t.Compose(
    [
        t.ToTensor(), # Converts np.array (H,W) or (H,W,C) to (C,H,W) tensor and scales to [0,1]
        # Normalize will be handled by PneumoniaDataset using loaded mean/std
        t.RandomAffine(degrees=(-5, 5), translate=(0.0, 0.05), scale=(0.9, 1.1)),
        t.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
    ]
)

default_val_transforms = t.Compose(
    [
        t.ToTensor(),
        # Normalize will be handled by PneumoniaDataset using loaded mean/std
        t.Resize(256, antialias=True), # Resize smaller edge to 256
        t.CenterCrop(224), # Crop center 224x224
    ]
)

def process_and_save_dicom_image(
    dicom_path: Path,
    output_npy_path: Path,
    resize_dim: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Reads a DICOM file, processes it, saves as .npy, and returns the processed array.

    Args:
        dicom_path (Path): Path to the input DICOM file.
        output_npy_path (Path): Path to save the processed .npy file.
        resize_dim (Tuple[int, int]): Target dimensions (height, width) for resizing.

    Returns:
        Optional[np.ndarray]: The processed image array (float16) or None if processing failed.
    """
    try:
        dicom_data = pydicom.dcmread(str(dicom_path))
        pixel_array = dicom_data.pixel_array
        if pixel_array.max() > 255: 
            pixel_array = np.clip(pixel_array, 0, 255)
        normalized_array = pixel_array / 255.0
        resized_array = cv2.resize(normalized_array, (resize_dim[1], resize_dim[0]), interpolation=cv2.INTER_LINEAR)
        resized_array_float16 = resized_array.astype(np.float16)
        output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy_path, resized_array_float16)
        return resized_array_float16
    except Exception as e:
        print(f"Error processing DICOM file {dicom_path}: {e}")
        return None


class PneumoniaDataset(Dataset):
    """
    Dataset for loading pneumonia images stored as .npy files.
    Normalization is applied internally using mean/std loaded from dataset_stats.json.
    """
    def __init__(
        self,
        root_dir: Union[str, Path], 
        split: str, 
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):
        self.data_root_dir = Path(root_dir) 
        self.split_dir = self.data_root_dir / split 
        self.transform = transform

        stats_path = self.data_root_dir / "dataset_stats.json"
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
            # print(f"Loaded dataset stats for '{split}' split: mean={self.mean:.4f}, std={self.std:.4f}")
        except FileNotFoundError:
            print(f"Warning: dataset_stats.json not found at {stats_path} for split '{split}'. Using default mean=0.5, std=0.25.")
            self.mean = 0.5 
            self.std = 0.25 
        except KeyError:
            print(f"Warning: 'mean' or 'std' key not found in {stats_path} for split '{split}'. Using default mean=0.5, std=0.25.")
            self.mean = 0.5
            self.std = 0.25
        
        self.normalize_transform = t.Normalize(self.mean, self.std)

        if dataframe.empty:
            self.samples = []
            print(f"Warning: DataFrame is empty for PneumoniaDataset (split: {split}). No samples will be loaded.")
            return

        self.patient_id_to_label = self._create_patient_id_map(dataframe)
        if not self.patient_id_to_label:
             print(f"Warning: Patient ID to label map is empty for split '{split}'. Check dataframe content and format.")

        self.samples = self._discover_samples(self.split_dir, self.patient_id_to_label)

        if not self.samples:
            print(f"Warning: No samples found in {self.split_dir} for '{split}' split. ")
        # else:
            # print(f"Initialized PneumoniaDataset for '{split}' split: Found {len(self.samples)} samples in {self.split_dir}.")

    @staticmethod
    def _create_patient_id_map(dataframe: pd.DataFrame) -> Dict[str, int]:
        if dataframe.empty or 'patientId' not in dataframe.columns or 'Target' not in dataframe.columns:
            # print("Warning: DataFrame is empty or 'patientId'/'Target' columns are missing for _create_patient_id_map.")
            return {}
        return dict(zip(dataframe["patientId"].astype(str), dataframe["Target"].astype(int)))

    @staticmethod
    def _discover_samples(scan_dir: Path, patient_id_to_label: Dict[str, int]) -> List[Tuple[Path, int]]:
        samples = []
        if not scan_dir.exists():
            # print(f"Warning: Scan directory {scan_dir} does not exist in _discover_samples.")
            return samples
        for label_subdir_name in ["0", "1"]:
            label_dir = scan_dir / label_subdir_name
            if not label_dir.is_dir():
                continue
            for file_path in label_dir.rglob("*.npy"):
                patient_id = file_path.stem
                if patient_id in patient_id_to_label:
                    label = patient_id_to_label[patient_id]
                    samples.append((file_path, label))
        # if not samples:
            # print(f"Warning: No .npy files linked to CSV patient IDs found in subdirectories of {scan_dir}.")
        samples.sort()
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_file(file_path: Path) -> np.ndarray:
        try:
            return np.load(file_path, allow_pickle=False).astype(np.float32)
        except Exception as e:
            # print(f"ERROR: Could not load file {file_path}. Reason: {e}")
            raise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < len(self.samples)):
            raise IndexError(f"Index {idx} out of bounds for PneumoniaDataset with length {len(self.samples)}")

        file_path, label = self.samples[idx]
        image_data_np = self._load_file(file_path)
        
        if self.transform:
            image_tensor = self.transform(image_data_np)
        else: 
            if image_data_np.ndim == 2: 
                image_tensor = torch.from_numpy(image_data_np).unsqueeze(0) 
            elif image_data_np.ndim == 3 and image_data_np.shape[0]==1: 
                 image_tensor = torch.from_numpy(image_data_np)
            elif image_data_np.ndim == 3 and image_data_np.shape[-1]==1: 
                 image_tensor = torch.from_numpy(image_data_np).permute(2,0,1)
            else:
                raise ValueError(f"Unsupported image dimensions: {image_data_np.shape}.")

        if isinstance(image_tensor, torch.Tensor):
            image_tensor = self.normalize_transform(image_tensor)
        else:
            raise TypeError(f"Data item at index {idx} (path: {file_path}) after initial transform was not a torch.Tensor. Got {type(image_tensor)}.")

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return image_tensor, label_tensor


def create_weighted_sampler(input_dataset: Dataset) -> Optional[WeightedRandomSampler]:
    if not hasattr(input_dataset, 'samples') or not input_dataset.samples:
        # print("Warning: input_dataset has no 'samples' attribute or 'samples' is empty. Cannot create sampler.")
        return None
    try:
        labels: List[int] = [sample_info[1] for sample_info in input_dataset.samples]
    except (TypeError, IndexError) as e:
        # print(f"Error extracting labels for weighted sampler: {e}. Ensure dataset.samples contains (path, label) tuples.")
        return None
    if not labels:
        # print("Warning: No labels found in dataset.samples. Cannot create sampler.")
        return None
    class_counts: Counter[int] = Counter(labels)
    if not class_counts or len(class_counts) == 0 or all(count == 0 for count in class_counts.values()):
        # print(f"Warning: Class counts are problematic for weighting: {class_counts}. Cannot create sampler.")
        return None
    num_samples_in_dataset: int = len(labels)
    sample_weights: torch.Tensor = torch.zeros(num_samples_in_dataset, dtype=torch.float32)
    for i, label_for_sample_i in enumerate(labels):
        if class_counts[label_for_sample_i] > 0:
            sample_weights[i] = 1.0 / class_counts[label_for_sample_i]
    if torch.sum(sample_weights).item() == 0:
        # print("Error: Sum of all sample weights is zero. Cannot create WeightedRandomSampler.")
        return None
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples_in_dataset,
        replacement=True
    )
    # print(f"WeightedRandomSampler created. Class counts: {class_counts}. Total samples to draw: {num_samples_in_dataset}.")
    return sampler

def create_dataloaders(
    data_dir: Union[str, Path],
    label_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders.

    Args:
        data_dir (Union[str, Path]): Path to the base directory of processed data (e.g., "Processed/").
                                     This directory should contain 'dataset_stats.json' and subdirs 'train/' and 'val/'.
        label_df (pd.DataFrame): DataFrame containing 'patientId' and 'Target' columns.
                                 Used by PneumoniaDataset for mapping patient IDs to labels.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for loading data.
        train_transform (Optional[Callable]): Transform to apply to training data. 
                                              Defaults to `default_train_transforms` from this module.
        val_transform (Optional[Callable]): Transform to apply to validation data.
                                            Defaults to `default_val_transforms` from this module.
        use_weighted_sampler (bool): If True, uses WeightedRandomSampler for the training DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and validation DataLoader.
    """
    data_dir = Path(data_dir)

    # Use default transforms if none are provided
    selected_train_transform = train_transform if train_transform is not None else default_train_transforms
    selected_val_transform = val_transform if val_transform is not None else default_val_transforms

    # Create training dataset
    train_dataset = PneumoniaDataset(
        root_dir=data_dir,
        split="train",
        dataframe=label_df,
        transform=selected_train_transform
    )

    # Create validation dataset
    val_dataset = PneumoniaDataset(
        root_dir=data_dir,
        split="val",
        dataframe=label_df,
        transform=selected_val_transform
    )

    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        if len(train_dataset) > 0: # Check if dataset is not empty
            train_sampler = create_weighted_sampler(train_dataset)
            if train_sampler:
                shuffle_train = False # Sampler handles shuffling
            else:
                print("Warning: Could not create weighted sampler for training. Using shuffle=True.")
        else:
            print("Warning: Training dataset is empty. Cannot create weighted sampler.")


    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Validation loader should not shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train DataLoader: {len(train_loader.dataset)} samples, Batch size: {batch_size}, Sampler: {'Weighted' if train_sampler else 'None'}, Shuffle: {shuffle_train}")
    print(f"Validation DataLoader: {len(val_loader.dataset)} samples, Batch size: {batch_size}, Shuffle: False")

    return train_loader, val_loader
