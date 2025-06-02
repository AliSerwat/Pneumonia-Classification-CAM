import os
from pathlib import Path
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torchvision import transforms as t
from typing import Tuple, Any, Optional, List, Dict, Callable
from collections import Counter
from tqdm.auto import tqdm

# Assuming utils.py is in the same package directory
# from .utils import DEVICE # Or handle device directly if preferred

# Default mean and std, can be overridden or calculated
DEFAULT_MEAN = 0.49072849471859337
DEFAULT_STD = 0.24792789306246266

# ============================  Transformations ============================ #

def get_train_transforms(img_size: int = 224, mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD) -> Callable:
    """
    Returns the default training transformations.

    Args:
        img_size (int): Target image size (height and width).
        mean (float): Mean for normalization.
        std (float): Standard deviation for normalization.

    Returns:
        Callable: A torchvision.transforms.Compose object.
    """
    return t.Compose([
        t.ToTensor(),
        t.Normalize(mean, std),
        t.RandomAffine(
            degrees=(-5, 5),
            translate=(0.0, 0.05),
            scale=(0.9, 1.1),
        ),
        t.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
    ])

def get_val_transforms(mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD) -> Callable:
    """
    Returns the default validation transformations.

    Args:
        mean (float): Mean for normalization.
        std (float): Standard deviation for normalization.

    Returns:
        Callable: A torchvision.transforms.Compose object.
    """
    return t.Compose([
        t.ToTensor(),
        t.Normalize(mean, std),
    ])

# ============================ Dataset Preprocessing ============================ #

def normalize_and_save_images(
    root_dicom_path: Path,
    save_processed_path: Path,
    labels_df: pd.DataFrame,
    target_img_size: int = 256,
    train_split_ratio: float = 0.8,
) -> Tuple[float, float]:
    """
    Processes DICOM images: normalizes, resizes, saves as .npy files,
    and calculates the global mean and standard deviation of pixel values
    for the training subset.

    Images are split into 'train' and 'val' subdirectories within `save_processed_path`
    based on `train_split_ratio`.

    Args:
        root_dicom_path (Path): Path to the directory containing raw DICOM image files.
        save_processed_path (Path): Path to the base directory where processed .npy images will be saved.
                                  Subdirectories 'train/0', 'train/1', 'val/0', 'val/1' will be created.
        labels_df (pd.DataFrame): Pandas DataFrame containing 'patientId' and 'Target' columns.
        target_img_size (int): The size (height and width) to resize images to.
        train_split_ratio (float): Fraction of images to allocate to the training set.

    Returns:
        Tuple[float, float]:
            - mean (float): The calculated global mean pixel value of the training images.
            - std (float): The calculated global standard deviation of pixel values
                           of the training images.
    """
    df_internal = labels_df.copy()
    if not isinstance(df_internal.index, pd.RangeIndex) or not (
        df_internal.index.start == 0 and df_internal.index.step == 1 and df_internal.index.stop == len(df_internal)
    ):
        print(
            "ℹ️ NOTICE: The DataFrame's index was not a standard sequential RangeIndex (0 to N-1)."
            " A local copy with a reset index will be used internally."
        )
        df_internal.reset_index(drop=True, inplace=True)

    total_sum_of_pixels: np.float64 = np.float64(0.0)
    total_sum_of_squared_pixels: np.float64 = np.float64(0.0)
    total_pixel_count: int = 0

    num_total_images = len(df_internal["patientId"])
    num_train: int = int(np.floor(train_split_ratio * num_total_images))

    try:
        if not df_internal["patientId"].is_unique:
            print("⚠️ WARNING: 'patientId' column contains duplicate values. Label lookup might be ambiguous.")
        patient_id_to_label: Dict[str, int] = df_internal.set_index("patientId")["Target"].to_dict()
    except KeyError:
        print("❌ ERROR: 'patientId' or 'Target' column not found in DataFrame. Cannot create label lookup.")
        return 0.0, 0.0

    print(f"⏳ Processing {num_total_images} images. Designated training set size: {num_train} images.")

    for i, patient_id in tqdm(
        enumerate(df_internal["patientId"]), total=num_total_images, desc="Processing DICOMs"
    ):
        dicom_file_path: Path = root_dicom_path / f"{patient_id}.dcm"
        try:
            pixel_array = pydicom.dcmread(str(dicom_file_path)).pixel_array
            normalized_pixel_array = pixel_array.astype(np.float32).clip(0, 255) / 255.0
            resized_pixel_array = cv2.resize(
                normalized_pixel_array, (target_img_size, target_img_size)
            ).astype(np.float16)
        except Exception as e:
            print(f"⚠️ Warning: Could not process DICOM {dicom_file_path} (ID: {patient_id}): {e}. Skipping.")
            continue

        try:
            label: int = patient_id_to_label[patient_id]
        except KeyError:
            print(f"⚠️ Warning: PatientID {patient_id} not found in label mapping. Skipping image.")
            continue

        label_str: str = str(label)
        train_or_val: str = "train" if i < num_train else "val"

        current_saving_dir: Path = save_processed_path / train_or_val / label_str
        current_saving_dir.mkdir(parents=True, exist_ok=True)
        np.save(current_saving_dir / f"{patient_id}.npy", resized_pixel_array)

        if train_or_val == "train":
            total_sum_of_pixels += np.sum(resized_pixel_array.astype(np.float64))
            total_sum_of_squared_pixels += np.sum(np.square(resized_pixel_array.astype(np.float64)))
            total_pixel_count += resized_pixel_array.size

    if total_pixel_count == 0:
        print("⚠️ Warning: No training images processed. Mean and Std cannot be computed.")
        mean, std = 0.0, 0.0
    else:
        mean = float(total_sum_of_pixels / total_pixel_count)
        variance: float = float((total_sum_of_squared_pixels / total_pixel_count) - (mean**2))
        if variance < 0:
            print(f"⚠️ Warning: Calculated variance is negative ({variance:.8f}). Clamping to 0.")
            variance = 0.0
        std = float(np.sqrt(variance))

    print(f"✅ Preprocessing complete. Processed images saved to '{save_processed_path}'.")
    print(f"📊 Global Training Mean (μ): {mean:.6f}")
    print(f"📊 Global Training Std (σ): {std:.6f}")
    print(f"🔢 Total pixels in training set for stats: {total_pixel_count}")

    return mean, std

# ============================ PyTorch Dataset Class ============================ #

class PneumoniaDataset(Dataset):
    """PyTorch Dataset for loading preprocessed .npy pneumonia CXR images."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processed_root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        dataset_split: str = "train", # e.g., "train" or "val"
    ):
        """
        Initializes the dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing 'patientId' and 'Target' columns.
            processed_root_dir (Union[str, Path]): Root directory where preprocessed .npy files
                                           are stored (e.g., .../Processed/train or .../Processed/val).
                                           This path should point to the specific split's directory.
            transform (Optional[Callable]): Optional transform to be applied on a sample.
            dataset_split (str): Informational string indicating the dataset split.
        """
        self.processed_root_dir = Path(processed_root_dir)
        self.transform = transform
        self.dataset_split = dataset_split

        self.patient_id_to_label = self._create_patient_id_map(dataframe)
        self.samples = self._discover_samples(self.processed_root_dir, self.patient_id_to_label)

        if not self.samples:
            print(f"Warning: No samples found for split '{dataset_split}' in directory '{self.processed_root_dir}'. "
                  f"Ensure this directory contains subdirectories '0' and '1' with .npy files, "
                  f"and that `normalize_and_save_images` has been run.")

    @staticmethod
    def _create_patient_id_map(dataframe: pd.DataFrame) -> Dict[str, int]:
        """Creates a dictionary mapping patientId to Target label."""
        if "patientId" not in dataframe.columns or "Target" not in dataframe.columns:
            raise ValueError("DataFrame must contain 'patientId' and 'Target' columns.")
        return dict(zip(dataframe["patientId"].astype(str), dataframe["Target"].astype(int)))

    @staticmethod
    def _discover_samples(scan_dir: Path, patient_id_to_label: Dict[str, int]) -> List[Tuple[Path, int]]:
        """
        Discovers .npy files and associates them with labels.
        Assumes scan_dir points to a directory like '.../train' or '.../val',
        which contains subdirectories '0' and '1' for labels.
        """
        samples = []
        for label_dir_name in ["0", "1"]: # Iterate through expected label subdirectories
            label_subdir = scan_dir / label_dir_name
            if not label_subdir.is_dir():
                # print(f"Info: Label directory '{label_subdir}' not found. Skipping.")
                continue

            for file_path in label_subdir.glob("*.npy"):
                patient_id = file_path.stem
                if patient_id in patient_id_to_label:
                    label = patient_id_to_label[patient_id]
                    if str(label) == label_dir_name: # Ensure file is in correct label subdir
                        samples.append((file_path, label))
                    # else:
                        # print(f"Warning: File {file_path} for patient {patient_id} has label {label} "
                        #       f"but was found in directory for label {label_dir_name}. Check data organization.")
                # else:
                    # print(f"Warning: Patient ID {patient_id} from file {file_path} not found in provided dataframe mapping.")
        samples.sort()
        return samples

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    @staticmethod
    def _load_file(file_path: Path) -> np.ndarray:
        """Loads a .npy file into a NumPy array (float32)."""
        try:
            return np.load(file_path, allow_pickle=True).astype(np.float32)
        except FileNotFoundError:
            print(f"ERROR: File not found at {file_path}")
            raise
        except Exception as e:
            print(f"ERROR: Could not load file {file_path}. Reason: {e}")
            raise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches the sample at the given index and applies transformations."""
        if not (0 <= idx < len(self.samples)):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.samples)} samples.")

        file_path, label = self.samples[idx]
        image_data_np = self._load_file(file_path) # Should be (H, W) or (C, H, W)

        if self.transform:
            # ToTensor expects (H, W, C) or (H, W). If (C,H,W) convert to (H,W,C) if C=1 or C=3
            # If image_data_np is (H,W), ToTensor adds channel dim.
            # If it's (C,H,W) and C=1, squeeze then ToTensor handles it.
            # If it's (C,H,W) and C=3, permute to (H,W,C) before ToTensor.
            if image_data_np.ndim == 3 and image_data_np.shape[0] == 1: # (1, H, W)
                 image_data_np = image_data_np.squeeze(0) # -> (H, W)
            elif image_data_np.ndim == 3 and image_data_np.shape[0] == 3: # (3, H, W)
                 image_data_np = image_data_np.transpose(1, 2, 0) # -> (H, W, 3)

            image_tensor = self.transform(image_data_np)
        else: # Manual conversion if no transform
            if image_data_np.ndim == 2: # Grayscale (H, W)
                image_tensor = torch.from_numpy(image_data_np).unsqueeze(0) # (1, H, W)
            elif image_data_np.ndim == 3: # (C, H, W) or (H, W, C)
                if image_data_np.shape[-1] in [1, 3]: # H, W, C
                    image_tensor = torch.from_numpy(image_data_np).permute(2, 0, 1) # C, H, W
                elif image_data_np.shape[0] in [1, 3]: # C, H, W
                    image_tensor = torch.from_numpy(image_data_np)
                else:
                    raise ValueError(f"Unhandled 3D shape for manual tensor conversion: {image_data_np.shape}")
            else:
                raise ValueError(f"Unexpected image dims: {image_data_np.shape}")

        if not isinstance(image_tensor, torch.Tensor): # Should be handled by ToTensor
            raise TypeError(f"Image data not converted to Tensor. Type: {type(image_tensor)}")

        label_tensor = torch.tensor(label, dtype=torch.float32) # For BCEWithLogitsLoss
        return image_tensor, label_tensor

# ============================ Data Balancing Utilities ============================ #

def get_balanced_dataset_indices(
    df: pd.DataFrame, ratio: int, target_column: str, id_column: str
) -> List[str]:
    """
    Identifies patient IDs for a class-balanced subset based on a given ratio.
    This function was modified to return a list of IDs to be used for filtering.

    Args:
        df: The input pandas DataFrame.
        ratio: The ratio of negative to positive samples.
        target_column: The name of the column containing the target labels (0 and 1).
        id_column: The name of the column containing unique identifiers.

    Returns:
        A list of patient IDs for the balanced dataset.
    """
    if not all(col in df.columns for col in [target_column, id_column]):
        raise ValueError(f"Missing required columns. Ensure '{target_column}' and '{id_column}' are in DataFrame.")

    positive_ids: List[str] = df.loc[df[target_column] == 1, id_column].sample(frac=1).tolist()
    negative_ids: List[str] = df.loc[df[target_column] == 0, id_column].sample(frac=1).tolist()

    if ratio <= 0:
        raise ValueError("Ratio must be a positive integer.")

    positive_count = len(positive_ids)
    negative_count = len(negative_ids)

    if positive_count == 0:
        print("Warning: No positive samples found. Returning all negative samples or empty list if none.")
        return negative_ids # Or [] if no negatives either

    # Determine the number of positive samples to use
    # If negative samples are the limiting factor for the ratio
    num_positive_to_use = min(positive_count, negative_count // ratio if ratio > 0 else negative_count)
    num_negative_to_use = num_positive_to_use * ratio

    selected_positive_ids = positive_ids[:num_positive_to_use]
    selected_negative_ids = negative_ids[:num_negative_to_use]

    balanced_id_list = selected_positive_ids + selected_negative_ids
    random.shuffle(balanced_id_list) # Shuffle the final list

    print(f"Balancing: Using {len(selected_positive_ids)} positive and {len(selected_negative_ids)} negative samples.")
    return balanced_id_list


def create_weighted_sampler(dataset: PneumoniaDataset) -> Optional[WeightedRandomSampler]:
    """
    Creates a WeightedRandomSampler for a PneumoniaDataset to handle class imbalance.
    Assumes `dataset.samples` is a list of (item_info, class_label) tuples.
    """
    try:
        labels: List[int] = [sample_info[1] for sample_info in dataset.samples]
    except AttributeError:
        print("ERROR: `dataset` is missing the `.samples` attribute.")
        raise
    except (TypeError, IndexError) as e:
        print(f"ERROR: Items in `dataset.samples` have unexpected structure: {e}.")
        raise

    if not labels:
        print("WARNING: No labels found in `dataset.samples`. Cannot create weighted sampler.")
        return None

    class_counts: Counter[int] = Counter(labels)
    if not class_counts: # Should be caught by `if not labels` but as a safeguard
        print("WARNING: Class counts are empty. Cannot create weighted sampler.")
        return None

    # Calculate weight for each sample: 1.0 / (Number of samples in that sample's class)
    # Ensure all labels present in `labels` are in `class_counts` keys for safety
    sample_weights_list = []
    for label in labels:
        count = class_counts.get(label)
        if count is None or count == 0:
            # This case should ideally not happen if labels come from class_counts
            print(f"Warning: Label {label} found in dataset but has zero count in class_counts. Assigning weight 0.")
            sample_weights_list.append(0.0)
        else:
            sample_weights_list.append(1.0 / count)

    sample_weights = torch.tensor(sample_weights_list, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


if __name__ == "__main__":
    import shutil # Moved import here as it's only used in __main__
    print("Testing data_loader.py components...")

    # Create a dummy labels DataFrame
    num_samples = 100
    patient_ids = [f"patient_{i:03d}" for i in range(num_samples)]
    targets = np.random.randint(0, 2, num_samples)
    dummy_labels_df = pd.DataFrame({"patientId": patient_ids, "Target": targets})

    # Setup dummy paths
    # project_root = Path(__file__).resolve().parent.parent # pneumonia_cam_project
    # base_test_path = project_root / "test_data_loader_output"
    # For running directly, use Path.cwd()
    base_test_path = Path.cwd() / "test_data_loader_output"

    dummy_dicom_root = base_test_path / "raw_dicoms"
    dummy_processed_root = base_test_path / "processed_data"

    shutil.rmtree(base_test_path, ignore_errors=True)
    dummy_dicom_root.mkdir(parents=True, exist_ok=True)
    dummy_processed_root.mkdir(parents=True, exist_ok=True)

    print(f"Created test directories at: {base_test_path}")

    # Create dummy DICOM files (just empty files for path testing)
    for pid in patient_ids:
        with open(dummy_dicom_root / f"{pid}.dcm", "w") as f:
            f.write("dummy dicom") # pydicom.dcmread will fail, but normalize_and_save_images handles exceptions

    print(f"Attempting to run normalize_and_save_images (will show errors for dummy DICOMs)...")
    # This will print errors because the dummy DICOMs are not valid,
    # but it tests the directory creation and stat calculation logic.
    mean_stat, std_stat = normalize_and_save_images(
        dummy_dicom_root, dummy_processed_root, dummy_labels_df, target_img_size=32
    )
    print(f"Calculated stats from dummy run: mean={mean_stat}, std={std_stat}")
    print(f"Check '{dummy_processed_root}' for 'train'/'val' and '0'/'1' subdirectories with .npy files.")


    # Test PneumoniaDataset
    # We need to create some dummy .npy files based on the output of normalize_and_save_images
    # Let's assume some files were created for 'train' split
    train_processed_path = dummy_processed_root / "train"
    # Manually create some dummy .npy files for testing PneumoniaDataset
    # This part needs actual .npy files for PneumoniaDataset to load.
    # For simplicity in this test, we'll check if PneumoniaDataset handles missing files.

    print(f"\nTesting PneumoniaDataset on train split: {train_processed_path}")
    train_dataset_transforms = get_train_transforms(img_size=32, mean=mean_stat if mean_stat!=0 else DEFAULT_MEAN, std=std_stat if std_stat!=0 else DEFAULT_STD)

    # Filter dummy_labels_df for patient IDs that would be in the 'train' split
    # (first 80% of patients if normalize_and_save_images was successful)
    num_train_samples_expected = int(0.8 * num_samples)
    train_patients_df = dummy_labels_df.iloc[:num_train_samples_expected]

    # Create dummy .npy files for the training set based on train_patients_df
    for _, row in train_patients_df.iterrows():
        pid = row["patientId"]
        label = str(row["Target"])
        label_dir = train_processed_path / label
        label_dir.mkdir(parents=True, exist_ok=True)
        dummy_npy_array = np.random.rand(32, 32).astype(np.float16) # Example, should match target_img_size
        np.save(label_dir / f"{pid}.npy", dummy_npy_array)

    train_dataset = PneumoniaDataset(
        dataframe=dummy_labels_df, # Full df for ID mapping
        processed_root_dir=train_processed_path, # Path to 'train'
        transform=train_dataset_transforms,
        dataset_split="train"
    )

    if len(train_dataset) > 0:
        print(f"Train dataset created with {len(train_dataset)} samples.")
        img, lbl = train_dataset[0]
        print(f"First sample - Image shape: {img.shape}, Label: {lbl}")

        # Test WeightedRandomSampler
        sampler = create_weighted_sampler(train_dataset)
        if sampler:
            print("WeightedRandomSampler created.")
            # Test DataLoader
            train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
            for batch_img, batch_lbl in train_loader:
                print(f"Batch - Image shape: {batch_img.shape}, Label shape: {batch_lbl.shape}")
                break
    else:
        print("Train dataset is empty, possibly due to dummy DICOM processing issues or path mismatches.")

    # Test get_balanced_dataset_indices
    print("\nTesting get_balanced_dataset_indices...")
    try:
        balanced_ids = get_balanced_dataset_indices(dummy_labels_df, ratio=1, target_column="Target", id_column="patientId")
        print(f"Number of balanced IDs (1:1 ratio): {len(balanced_ids)}")
        if balanced_ids:
            balanced_df_example = dummy_labels_df[dummy_labels_df["patientId"].isin(balanced_ids)]
            print(f"Example balanced df target counts:\n{balanced_df_example['Target'].value_counts()}")
    except ValueError as e:
        print(f"Error in get_balanced_dataset_indices: {e}")


    # Clean up
    # print(f"\nCleaning up test directory: {base_test_path}")
    # shutil.rmtree(base_test_path, ignore_errors=True)
    print("\n--- data_loader.py tests finished ---")
    print(f"NOTE: For full testing of normalize_and_save_images, valid DICOM files are needed at '{dummy_dicom_root}'.")
    print(f"The dummy .npy files for PneumoniaDataset testing were created in '{train_processed_path}'.")
    print(f"Consider manually cleaning up '{base_test_path}' if needed.")
