# %% id="62e0838c"
# ============================ 📦 Standard Libraries ============================ #
# Environment config (e.g., for DDP or log levels)
import random
import os
import gc  # Garbage collection
import math  # Mathematical operations
import json  # JSON file handling
import time  # Timing utilities
import shutil  # File operations
import random  # Random operations
import logging  # Logging setup
import subprocess  # Subprocess management
import concurrent.futures  # Parallel processing
from pathlib import Path  # Path operations
from collections import Counter  # Frequency counter
import warnings  # Warning handling

# ============================ 📊 Data Handling & Visualization ============================ #
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computation
import matplotlib.pyplot as plt  # Plotting
import cv2  # OpenCV for image processing
import pydicom  # DICOM medical image handling

# ============================ 📁 PyTorch Core ============================ #
import torch  # Core PyTorch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import onnx  # Neural network layers
import torch.nn.functional as F  # Functional operations
import torch.optim as optim  # Optimizers
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
)  # Dataset & loaders
from torch.cuda.amp import autocast, GradScaler  # Mixed precision

# ============================ 🧠 PyTorch Utilities ============================ #
from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
)  # Learning rate schedulers

# ============================ 📈 Metrics & Summaries ============================ #
import torchmetrics  # Metric utilities
from torchmetrics import Accuracy, AUROC  # Specific metrics
from torchinfo import summary  # Model summary

# ============================ 🧪 PyTorch Lightning ============================ #
# import pytorch_lightning as pl # Commented out as not used in the final script
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import (
#     ModelCheckpoint,
#     EarlyStopping,
#     LearningRateMonitor,
#     GradientAccumulationScheduler,
#     StochasticWeightAveraging,
#     BackboneFinetuning,
# )

# ============================ 🧰 TorchVision & Timm ============================ #
import timm  # Pretrained models
import torchvision  # Vision dataset & models
from torchvision import datasets, transforms as t  # Image transforms

# ============================ 💻 Notebook Utilities ============================ #
from IPython.display import display, HTML  # Jupyter display utilities
# from tqdm.notebook import tqdm  # Progress bars (Jupyter) # Replaced with tqdm.auto
from tqdm.auto import tqdm as tqdm_auto  # General progress bars

# ============================ 🔤 Typing Support ============================ #
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ============================⚠️ Miscellaneous ============================ #
# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress TensorFlow logs (if used indirectly)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %% [markdown]
# ---
# * **Device Detection (`torch.device(...)`):** This PyTorch function automatically identifies if a CUDA-enabled GPU is available and sets the `device` variable accordingly, allowing for hardware acceleration if possible.
# ---

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ---
# # 💻 Kaggle Environment Configuration
# (Content removed for brevity in this prompt, but was in original)
# ---

# %% id="dcfffa97"
# Kaggle specific path setup (commented out for general use)
# dataset_parent = Path.home().parent / "kaggle" / "input"
# output_directory = Path.home().parent / "kaggle" / "working"
# num_workers = os.cpu_count()
# BATCH_SIZE = 64
# HOME = Path.home()
# BASE_PATH = dataset_parent / "rsna-pneumonia-detection-challenge"
# ROOT_PATH = BASE_PATH / "stage_2_train_images"
# SAVE_PATH = output_directory / "Processed"
# for directory in [(output_directory / "logs"), SAVE_PATH]:
#     shutil.rmtree(directory, ignore_errors=True)

# %% [markdown]
# -----
# # ⚙️ Configuration for Non-Kaggle Platforms
# (Content removed for brevity, but was in original)
# -----

# %%
# Setup for Kaggle API key (commented out for general use)
# home_dir = Path.home()
# kaggle_config_dir = home_dir / ".kaggle"
# os.makedirs(kaggle_config_dir, exist_ok=True)
# kaggle_json = {"username": "YOUR_KAGGLE_USERNAME", "key": "YOUR_KAGGLE_API_KEY"}
# kaggle_json_path = kaggle_config_dir / "kaggle.json"
# with open(kaggle_json_path, "w") as f:
#     json.dump(kaggle_json, f)
# os.chmod(kaggle_json_path, 0o600)

# %% colab={"base_uri": "https://localhost:8080/"} id="Y_GPQljRtivG" outputId="e48337db-f258-4978-c210-79c60e991043" language="bash"
# # Download the RSNA Pneumonia Detection Challenge dataset using Kaggle API
# # kaggle competitions download -c rsna-pneumonia-detection-challenge

# %% id="6aeb7519"
# Define consistent project paths for local execution
HOME = Path.home()
BASE_PATH = Path.cwd() / "Pneumonia"  # Main project data folder
ROOT_PATH = BASE_PATH / "stage_2_train_images"  # Raw DICOMs
SAVE_PATH = Path.cwd() / "Processed"  # For .npy files

# %%
# Ensure all necessary directories exist
for path in [BASE_PATH, ROOT_PATH, SAVE_PATH]:
    if not path.exists():
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)

zip_path = HOME / "rsna-pneumonia-detection-challenge.zip"

def is_directory_empty(directory_path: Path) -> bool:
    try:
        return not any(directory_path.iterdir())
    except FileNotFoundError:
        return True

if zip_path.exists():
    if is_directory_empty(BASE_PATH):
        print(f"Output directory '{BASE_PATH}' is empty. Proceeding with extraction.")
        try:
            print(f"Extracting {zip_path.name} to {BASE_PATH} ...")
            subprocess.run(["unzip", str(zip_path), "-d", str(BASE_PATH)], check=True)
            print("Unzipping completed successfully.")
        except FileNotFoundError:
            print("Error: 'unzip' command not found. Please ensure it's installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during unzipping: {e}")
    else:
        print(f"Warning: Output directory '{BASE_PATH}' is not empty.")
else:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        print(f"Downloading dataset to {HOME} ...")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "rsna-pneumonia-detection-challenge", "-p", str(HOME)],
            check=True,
        )
        if zip_path.exists():
            if is_directory_empty(BASE_PATH):
                print(f"Extracting {zip_path.name} to {BASE_PATH} ...")
                subprocess.run(["unzip", str(zip_path), "-d", str(BASE_PATH)], check=True)
                print("Unzipping completed successfully.")
            else:
                print(f"Warning: Output directory '{BASE_PATH}' is not empty.")
        else:
            print(f"Zip file not found at {zip_path}")
    except FileNotFoundError:
        print("Error: Kaggle CLI not found. Please install it.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")

# %% [markdown]
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 197} id="917b18fb" outputId="afa8c046-2fbc-4d8e-ffe7-619ff701d80b"
try:
    BASE_PATH
except NameError:
    BASE_PATH = Path("./")

all_items = list(BASE_PATH.iterdir())
directories = [item.name for item in all_items if item.is_dir()]
files = [item.name for item in all_items if item.is_file()]
html_output = ""
if directories:
    dir_html = '<span style="color:#6699CC; font-weight:bold; font-size:1.2em;">📁 Directories:</span><br>'
    for directory in sorted(directories):
        dir_html += f'<span style="color:#8FBC8F;">&nbsp;&nbsp;📂 {directory}</span><br>'
    html_output += dir_html
if directories and files:
    html_output += '<br><span style="color:#A9A9A9;">---</span><br>'
if files:
    file_html = '<span style="color:#6699CC; font-weight:bold; font-size:1.2em;">📄 Files:</span><br>'
    for file in sorted(files):
        file_html += f'<span style="color:#FFD700;">&nbsp;&nbsp;📄 {file}</span><br>'
    html_output += file_html
display(HTML(html_output))

# %% id="3063d53d"
labels_df_path = BASE_PATH / "stage_2_train_labels.csv"
if labels_df_path.exists():
    labels = pd.read_csv(labels_df_path)
else:
    print(f"Labels file not found at {labels_df_path}. Please ensure it's downloaded and extracted correctly.")
    labels = pd.DataFrame(columns=['patientId', 'Target']) # Placeholder

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="2b9253dc" outputId="bb040509-1505-4afa-8016-0e38540060c9"
if not labels.empty:
    print(labels.head())

# %% id="31f07b08"
if not labels.empty:
    relevant_columns = labels.columns.tolist()
    labels["nonnull_count"] = labels[relevant_columns].notnull().sum(axis=1)
    max_nonnull_index = labels.groupby("patientId")["nonnull_count"].idxmax()
    deduplicated_labels = labels.loc[max_nonnull_index].drop(columns="nonnull_count")
    deduplicated_labels.reset_index(drop=True, inplace=True)
else:
    deduplicated_labels = pd.DataFrame(columns=['patientId', 'Target'])


# %% id="e75d56cd"
import copy
labels = copy.deepcopy(deduplicated_labels)

# %% colab={"base_uri": "https://localhost:8080/", "height": 315} id="771bb8ee" outputId="71313fc7-8f6c-4308-ad6d-32e6df42f69b"
if not labels.empty:
    print("Handled potential label inconsistencies...
")
    print(f"Shape of the processed labels DataFrame: {labels.shape}

")
    print(labels.head())

# %% id="4a956cf7"
def get_balanced_dataset(
    df: pd.DataFrame, ratio: int, target_column: str, id_column: str
) -> pd.DataFrame | List:
    if df.empty: return pd.DataFrame()
    required_columns = {target_column, id_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    positive_ids: List[str] = (
        df.loc[df[target_column] == 1, id_column].sample(frac=1).tolist()
    )
    negative_ids: List[str] = (
        df.loc[df[target_column] == 0, id_column].sample(frac=1).tolist()
    )
    if ratio > 1:
        positive_count = len(positive_ids)
        negative_count = len(negative_ids)
        group_count = min(positive_count, negative_count // ratio)
        id_list: List[str] = []
        negative_index = 0
        for positive_index in range(group_count):
            id_list.append(positive_ids[positive_index])
            for _ in range(ratio):
                if negative_index < negative_count:
                    id_list.append(negative_ids[negative_index])
                    negative_index += 1
        return df[df[id_column].isin(id_list)].reset_index(drop=True)
    else:
        return df[target_column].drop_duplicates().sample(frac=1).tolist()

# %% id="3f884161"
if ROOT_PATH.exists():
    train_dcm_path = sorted(ROOT_PATH.rglob("*.dcm"))
else:
    print(f"ROOT_PATH {ROOT_PATH} does not exist. Skipping DICOM path search.")
    train_dcm_path = []

# %% id="bac744d3"
def visualize_dicom_images(
    train_dcm_paths: List[Path],
    labels_df: pd.DataFrame,
    grid_rows: int = 2,
    grid_cols: int = 2,
    dpi: int = 200,
) -> None:
    if not train_dcm_paths or labels_df.empty:
        print("No DICOM paths or labels to visualize.")
        return
    patient_id_to_label: Dict[str, int] = labels_df.set_index("patientId")[
        "Target"
    ].to_dict()
    max_train_idx = int(len(train_dcm_paths) * 0.8)
    if max_train_idx == 0 and len(train_dcm_paths) > 0: max_train_idx = 1
    elif max_train_idx == 0:
        print("Warning: No DICOM paths provided for visualization.")
        return
    fig, ax = plt.subplots(grid_rows, grid_cols, dpi=dpi, figsize=(grid_cols * 3, grid_rows * 3))
    axes_flat = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
    for i, current_ax in enumerate(axes_flat):
        if i >= max_train_idx : break # Ensure we don't pick beyond available images
        random_idx = np.random.randint(0, max_train_idx)
        dcm_path = train_dcm_paths[random_idx]
        try:
            dicom_data = pydicom.dcmread(str(dcm_path)).pixel_array
            resized_image = cv2.resize(dicom_data, (256, 256))
        except Exception as e:
            print(f"Error reading or processing DICOM file {dcm_path.name}: {e}")
            current_ax.axis("off")
            current_ax.set_title("Error loading image", fontsize=8, color="red")
            continue
        patient_id = dcm_path.stem
        label_val = patient_id_to_label.get(patient_id, -1)
        display_label = "Healthy" if label_val == 0 else "Pneumonia Patient" if label_val == 1 else "Unknown"
        title_color = "green" if label_val == 0 else "red" if label_val == 1 else "gray"
        current_ax.imshow(resized_image, cmap="bone")
        current_ax.axis("off")
        current_ax.set_title(f"ID: {patient_id}
{display_label}", fontsize=8, color=title_color)
    plt.tight_layout()
    plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 736} id="db9cfa1a" outputId="7624ba0e-6c3c-4da3-9b64-ad93cdb65344"
if train_dcm_path and not labels.empty:
    visualize_dicom_images(train_dcm_path, labels)

# %% [markdown] id="567fe3c6"
# ---
# ### ⚙️ Preprocessing ...
# (Content removed for brevity)
# ---

# %% id="99af5681"
def normalize_img(
    ROOT_PATH: Path,
    SAVE_PATH: Path,
    df: pd.DataFrame,
) -> Tuple[float, float]:
    if df.empty or not ROOT_PATH.exists():
        print("DataFrame is empty or ROOT_PATH does not exist. Skipping normalization.")
        return 0.0, 0.0

    df_internal = df.reset_index(drop=True) # Always work with a reset index copy

    total_sum_of_pixels: np.float64 = np.float64(0.0)
    total_sum_of_squared_pixels: np.float64 = np.float64(0.0)
    total_pixel_count: int = 0
    num_train: int = int(np.floor(0.8 * len(df_internal["patientId"])))

    if not df_internal["patientId"].is_unique:
        print("⚠️ WARNING: 'patientId' column contains duplicate values.")
    patient_id_to_label: Dict[str, int] = df_internal.set_index("patientId")["Target"].to_dict()

    print(f"⏳ Processing {len(df_internal['patientId'])} images. Training set size: {num_train} images.")

    for i, patientId_from_zip in tqdm_auto(
        zip(df_internal.index.values, df_internal["patientId"]),
        total=len(df_internal),
        desc="Processing images",
    ):
        current_patient_id: str = patientId_from_zip
        dicom_file_path: Path = ROOT_PATH / f"{current_patient_id}.dcm"
        try:
            pixel_array = pydicom.dcmread(str(dicom_file_path)).pixel_array
            normalized_pixel_array = pixel_array.clip(0, 255) / 255.0
            resized_pixel_array = cv2.resize(normalized_pixel_array, (256, 256)).astype(np.float16)
        except Exception as e:
            print(f"⚠️ Warning: Could not process {dicom_file_path}: {e}. Skipping.")
            continue
        try:
            label: int = patient_id_to_label[current_patient_id]
        except KeyError:
            print(f"⚠️ Warning: PatientID {current_patient_id} not in label map. Skipping.")
            continue
        label_str: str = str(label)
        train_or_val: str = "train" if i < num_train else "val"
        current_saving_dir: Path = SAVE_PATH / train_or_val / label_str
        current_saving_dir.mkdir(parents=True, exist_ok=True)
        np.save(current_saving_dir / f"{current_patient_id}.npy", resized_pixel_array)

        if train_or_val == "train":
            total_sum_of_pixels += np.sum(resized_pixel_array.astype(np.float64))
            total_sum_of_squared_pixels += np.sum(np.square(resized_pixel_array.astype(np.float64)))
            total_pixel_count += resized_pixel_array.size
    if total_pixel_count == 0:
        mean: float = 0.0
        std: float = 0.0
    else:
        mean = float(total_sum_of_pixels / total_pixel_count)
        variance: float = float((total_sum_of_squared_pixels / total_pixel_count) - (mean**2))
        if variance < 0: variance = 0.0
        std = float(np.sqrt(variance))
    print(f"✅ Preprocessing complete. μ: {mean:.6f}, σ: {std:.6f}, Pixels: {total_pixel_count}")
    return mean, std

# %% colab={"base_uri": "https://localhost:8080/", "height": 214} id="6c15c59f"
if not labels.empty and ROOT_PATH.exists() and SAVE_PATH.exists():
    mean, std = normalize_img(ROOT_PATH, SAVE_PATH, labels)
else:
    print("Skipping normalize_img due to missing labels, ROOT_PATH or SAVE_PATH.")
    mean, std = 0.4907, 0.2479 # Fallback to calculated values if processing skipped

# %% id="c23938a1"
train_transforms = t.Compose(
    [
        t.ToTensor(),
        t.Normalize(mean, std), # Use calculated or fallback mean/std
        t.RandomAffine(degrees=(-5, 5), translate=(0.0, 0.05), scale=(0.9, 1.1)),
        t.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    ]
)
val_transforms = t.Compose([t.ToTensor(), t.Normalize(mean, std)])

# %% id="62a3753a"
class PneumoniaDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, root_dir: Union[str, Path],
        transform: Optional[Callable] = None, dataset_split: str = "train"
    ):
        self.root_dir = Path(root_dir) / dataset_split # Path adjusted to include train/val
        self.transform = transform
        # self.dataset_split = dataset_split # Not directly used if root_dir includes split

        if dataframe.empty:
            self.samples = []
            print(f"Warning: DataFrame is empty for PneumoniaDataset (split: {dataset_split}).")
            return

        patient_id_to_label = self._create_patient_id_map(dataframe)
        self.samples = self._discover_samples(self.root_dir, patient_id_to_label)
        if not self.samples:
            print(f"Warning: No samples found in {self.root_dir} for {dataset_split} split.")


    @staticmethod
    def _create_patient_id_map(dataframe: pd.DataFrame) -> dict:
        if 'patientId' not in dataframe.columns or 'Target' not in dataframe.columns:
            # print("Warning: 'patientId' or 'Target' missing from dataframe for _create_patient_id_map.")
            return {}
        return dict(zip(dataframe["patientId"].astype(str), dataframe["Target"].astype(int)))

    @staticmethod
    def _discover_samples(scan_dir: Path, patient_id_to_label: dict) -> list:
        samples = []
        if not scan_dir.exists(): # Check if the specific split directory exists
            # print(f"Warning: Scan directory {scan_dir} does not exist in _discover_samples.")
            return samples
        for label_dir in scan_dir.iterdir(): # Iterate through '0' and '1' subdirs
            if label_dir.is_dir():
                expected_label_dir_str = label_dir.name
                for file_path in label_dir.rglob("*.npy"):
                    patient_id = file_path.stem
                    if patient_id in patient_id_to_label:
                        label = patient_id_to_label[patient_id]
                        if str(label) == expected_label_dir_str: # Check if parent dir name matches label
                             samples.append((file_path, label))
        samples.sort()
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_file(file_path: Path) -> np.ndarray:
        try:
            return np.load(file_path, allow_pickle=True).astype(np.float32)
        except Exception as e:
            # print(f"ERROR: Could not load file {file_path}. Reason: {e}")
            raise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < len(self.samples)):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.samples)}")
        file_path, label = self.samples[idx]
        image_data_np = self._load_file(file_path)
        if self.transform:
            image_tensor = self.transform(image_data_np)
        else:
            if image_data_np.ndim == 2: image_tensor = torch.from_numpy(image_data_np).unsqueeze(0)
            elif image_data_np.ndim == 3:
                if image_data_np.shape[-1] in [1,3,4]: image_tensor = torch.from_numpy(image_data_np).permute(2,0,1)
                else: image_tensor = torch.from_numpy(image_data_np)
            else: raise ValueError(f"Unsupported image dimensions: {image_data_np.shape}")
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"Image not converted to Tensor. Type: {type(image_tensor)}")
        label_tensor = torch.tensor(label, dtype=torch.float32) # BCEWithLogitsLoss expects float target
        return image_tensor, label_tensor

# %% id="0649f02e"
def create_weighted_sampler(input_dataset: Dataset) -> Optional[WeightedRandomSampler]:
    if not hasattr(input_dataset, 'samples') or not input_dataset.samples:
        # print("Dataset has no samples or .samples attribute for weighted sampler.")
        return None
    try:
        labels: List[int] = [sample_info[1] for sample_info in input_dataset.samples]
    except (TypeError, IndexError) as e:
        # print(f"Error extracting labels for weighted sampler: {e}")
        return None
    if not labels: return None
    class_counts: Counter[int] = Counter(labels)
    if not class_counts: return None # Should not happen if labels is not empty
    num_samples_in_dataset: int = len(labels)
    sample_weights: torch.Tensor = torch.empty(num_samples_in_dataset, dtype=torch.float32)
    for i in range(num_samples_in_dataset):
        current_label_for_sample_i: int = labels[i]
        if class_counts[current_label_for_sample_i] == 0: # Avoid division by zero
             sample_weights[i] = 0 # Or some other default/small weight
             # print(f"Warning: Class {current_label_for_sample_i} has zero count in weighted sampler.")
        else:
             weight_for_sample_i: float = 1.0 / class_counts[current_label_for_sample_i]
             sample_weights[i] = weight_for_sample_i
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples_in_dataset, replacement=True)
    return sampler

# %% id="cc59f801"
# Create the training and validation dataset
# Ensure SAVE_PATH points to the parent of 'train' and 'val' directories
if SAVE_PATH.exists() and not labels.empty:
    train_dataset = PneumoniaDataset(labels, SAVE_PATH, train_transforms, "train")
    val_dataset = PneumoniaDataset(labels, SAVE_PATH, val_transforms, "val")
else:
    print("SAVE_PATH does not exist or labels are empty. Cannot create datasets.")
    # Create dummy datasets to prevent errors if pipeline continues
    train_dataset = Dataset() # Placeholder
    val_dataset = Dataset()   # Placeholder

# %%
# def load_file(file_path): # This is already a static method in PneumoniaDataset
#     return np.load(file_path, allow_pickle=True).astype(np.float32)

# %%
BATCH_SIZE = 64 # Already defined earlier, ensure consistency
num_workers = os.cpu_count() # Already defined earlier

# %%
if hasattr(train_dataset, 'samples') and train_dataset.samples: # Check if dataset is not empty
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=create_weighted_sampler(train_dataset), shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
else:
    print("Train dataset is empty. Creating an empty DataLoader.")
    train_loader = DataLoader(Dataset()) # Empty DataLoader

if hasattr(val_dataset, 'samples') and val_dataset.samples: # Check if dataset is not empty
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
else:
    print("Validation dataset is empty. Creating an empty DataLoader.")
    val_loader = DataLoader(Dataset()) # Empty DataLoader


# %% id="b8ebd678"
def visualize_dataset_samples(
    dataset: Dataset, num_rows: int = 2, num_cols: int = 2, dpi: int = 200,
    class_names: Optional[Union[List[str], Dict[int, str]]] = None,
) -> None:
    if not isinstance(dataset, Dataset) or len(dataset) == 0:
        # print("Invalid or empty dataset for visualization.")
        return
    fig, axes = plt.subplots(num_rows, num_cols, dpi=dpi)
    axes_flat = [axes] if num_rows * num_cols == 1 else axes.flatten()
    num_to_show = min(num_rows * num_cols, len(dataset))
    try:
        random_indices = np.random.choice(len(dataset), size=num_to_show, replace=False)
    except ValueError as e:
        # print(f"Random sampling failed: {e}")
        return
    for i, idx in enumerate(random_indices):
        ax = axes_flat[i]
        try:
            image_tensor, label_data = dataset[idx]
            if not isinstance(image_tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(image_tensor)}")
            img = image_tensor.detach().cpu().numpy()
            if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)
            if img.ndim == 3:
                if img.shape[0] == 1: img = img.squeeze(0)
                elif img.shape[0] == 3: img = img.transpose(1, 2, 0)
            elif img.ndim not in (2,3): raise ValueError(f"Unsupported shape: {img.shape}")
            label_index = label_data.item() if isinstance(label_data, torch.Tensor) and label_data.numel()==1 else int(label_data)
            title_color, label_str = "darkblue", f"Label {label_index}"
            if class_names:
                if isinstance(class_names, list): label_str = class_names[label_index] if 0 <= label_index < len(class_names) else f"Idx {label_index}!"
                elif isinstance(class_names, dict): label_str = class_names.get(label_index, f"Idx {label_index}!")
            else:
                if label_index == 0: label_str, title_color = "Healthy", "green"
                elif label_index == 1: label_str, title_color = "Pneumonia", "red"
            ax.imshow(img, cmap="bone" if img.ndim == 2 else None)
            ax.set_title(f"Sample {idx} | {label_str}", fontsize=8, color=title_color)
        except Exception as e:
            # print(f"Error displaying sample {idx}: {e}")
            ax.text(0.5,0.5, f"Error
{type(e).__name__}", ha="center", va="center", color="red", fontsize=7, wrap=True)
        ax.axis("off")
    for j in range(num_to_show, len(axes_flat)): axes_flat[j].axis("off")
    fig.tight_layout()
    plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 690} id="b826caf3" outputId="a69492a5-2986-4200-b3be-d379d257a98e"
if hasattr(train_dataset, 'samples') and train_dataset.samples:
    visualize_dataset_samples(train_dataset)

# %% [markdown] id="006c1fe1"
# ---
# # 🧠 Decoding "GELU Gain" ...
# (Content removed for brevity)
# ---

# %% id="0c3895e1"
class MedicalAttentionGate(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        bottleneck_dimension = max(1, dimension // 4)
        if dimension < 4: print(f"⚠️ WARNING: Input dimension ({dimension}) small for MedicalAttentionGate.")
        self.attention_net = nn.Sequential(
            nn.Linear(dimension, bottleneck_dimension), nn.Sigmoid(),
            nn.Linear(bottleneck_dimension, dimension), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention_net(x)
        return x * attention_weights

# %% [markdown]
# ### Calling Class Methods Within a Class ...
# (Content removed for brevity)
# ```

# %% id="1142bc80"
class EnhancedClassifier(nn.Module):
    _EMPIRICAL_GELU_GAIN: Optional[float] = None
    @staticmethod
    def _get_empirical_gelu_gain() -> float:
        if EnhancedClassifier._EMPIRICAL_GELU_GAIN is None:
            with torch.no_grad():
                toy_input = torch.randn(100_000)
                activated_output = nn.functional.gelu(toy_input)
                std_dev_input = toy_input.std().item()
                std_dev_output = activated_output.std().item()
                if std_dev_output == 0: EnhancedClassifier._EMPIRICAL_GELU_GAIN = 1.0
                else: EnhancedClassifier._EMPIRICAL_GELU_GAIN = std_dev_input / std_dev_output
        return EnhancedClassifier._EMPIRICAL_GELU_GAIN

    def __init__(self, in_features: int, out_features: int,
                 dropout_rate: float = 0.3, hidden_dim: int = 512):
        super().__init__()
        hidden_dim_expanded = hidden_dim * 2
        self.feature_map_processor = nn.Sequential(
            nn.Linear(in_features, hidden_dim_expanded), nn.BatchNorm1d(hidden_dim_expanded),
            nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_expanded, hidden_dim),
            MedicalAttentionGate(hidden_dim), nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim, out_features),
        )
        self._initialize_module_weights()

    def _initialize_module_weights(self) -> None:
        custom_gelu_gain = EnhancedClassifier._get_empirical_gelu_gain()
        for module in self.feature_map_processor.modules():
            if isinstance(module, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in == 0: nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    std_deviation = custom_gelu_gain / math.sqrt(fan_in)
                    nn.init.normal_(module.weight, mean=0.0, std=std_deviation)
                if module.bias is not None: nn.init.constant_(module.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map_processor(x)

# %% [markdown] id="1f7c155e"
# ---
# ### ❓ Explain `list(self.backbone.children())` ...
# (Content removed for brevity)
# ---

# %% id="b1733ca6"
class PneumoniaModelCAM(nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int = 1, in_channels: int = 1,
                 criterion_pos_weight: float = 1.0, classifier_hidden_dim: int = 512,
                 classifier_dropout_rate: float = 0.3, img_size: int = 256): # Default img_size
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size # Store img_size

        try:
            self.backbone = timm.create_model(
                self.pretrained_model_name, pretrained=True, num_classes=0,
                global_pool="", in_chans=self.in_channels
            )
            # print(f"✅ Backbone '{self.pretrained_model_name}' created with `in_chans={self.in_channels}`.")
        except Exception as e_timm:
            # print(f"⚠️ WARNING: Timm failed for in_chans={self.in_channels}: {e_timm}. Trying fallback.")
            self.backbone = timm.create_model(
                self.pretrained_model_name, pretrained=True, num_classes=0, global_pool=""
            )
            if self.in_channels != 3 and not self._attempt_manual_first_conv_override(self.in_channels):
                print(f"❌ ERROR: Failed to adapt backbone for {self.in_channels} channels.")

        self.global_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_features = nn.Flatten(start_dim=1)

        # Determine num_backbone_out_features dynamically
        # Use the stored self.img_size
        dummy_input = torch.randn(2, self.in_channels, self.img_size, self.img_size)
        self.backbone.eval()
        with torch.no_grad():
            dummy_feature_maps = self.backbone(dummy_input)
        self.num_backbone_out_features = dummy_feature_maps.shape[1]
        self.backbone.train() # Set back to train mode

        self.classifier = EnhancedClassifier(
            in_features=self.num_backbone_out_features, out_features=self.num_classes,
            dropout_rate=classifier_dropout_rate, hidden_dim=classifier_hidden_dim
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(criterion_pos_weight, dtype=torch.float))

    def _attempt_manual_first_conv_override(self, target_in_channels: int) -> bool:
        # Simplified version of manual override
        first_conv_layer_name = None
        if hasattr(self.backbone, "conv_stem") and isinstance(self.backbone.conv_stem, nn.Conv2d):
            first_conv_layer_name = "conv_stem"
        elif hasattr(self.backbone, "conv1") and isinstance(self.backbone.conv1, nn.Conv2d):
            first_conv_layer_name = "conv1"

        if first_conv_layer_name:
            original_conv = getattr(self.backbone, first_conv_layer_name)
            # print(f"    Attempting to manually modify '{first_conv_layer_name}' for {target_in_channels} input channels.")
            new_conv = nn.Conv2d(
                target_in_channels, original_conv.out_channels,
                kernel_size=original_conv.kernel_size, stride=original_conv.stride,
                padding=original_conv.padding, bias=(original_conv.bias is not None)
            )
            # Copy weights if possible (e.g., average them if target_in_channels < original_conv.in_channels)
            if original_conv.weight.data.shape[1] == 3 and target_in_channels == 1: # RGB to Grayscale
                new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            elif original_conv.weight.data.shape[1] == target_in_channels : # if it was already compatible
                new_conv.weight.data = original_conv.weight.data
            # else: just use random init for new_conv

            setattr(self.backbone, first_conv_layer_name, new_conv)
            # print(f"    Manually replaced '{first_conv_layer_name}'.")
            return True
        # print(f"    ⚠️ Manual override: Could not find standard first Conv2d to modify.")
        return False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features_map = self.backbone(x)
        pooled_features = self.global_adaptive_pool(features_map)
        flat_features = self.flatten_features(pooled_features)
        logits = self.classifier(flat_features)
        # For ONNX export and BCEWithLogitsLoss, if num_classes is 1, squeeze last dim.
        # This is now handled based on self.num_classes which is static for export.
        if self.num_classes == 1:
            processed_logits = logits.squeeze(-1) # Squeeze the last dimension if it's 1
        else:
            processed_logits = logits
        return processed_logits, features_map

# %% id="6c7d2f5d"
def clear_memory(vars_to_delete=None):
    if vars_to_delete is None:
        vars_to_delete = ["model", "optimizer", "train_dataset", "val_dataset", "train_loader", "val_loader"]
    # print("🧹 Attempting to delete global variables...")
    deleted_vars_count = 0
    for var_name in vars_to_delete:
        if var_name in globals():
            del globals()[var_name]
            deleted_vars_count +=1
    # if deleted_vars_count > 0: print(f"✅ Deleted {deleted_vars_count} global variable(s).")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # print("✅ Memory clearing actions complete.")


# %%
# ---------------------------------------------------------------------------- #
#                      🐍 USER-DEFINED COMPONENTS (Critical) 🐍                  #
# ---------------------------------------------------------------------------- #
IMG_SIZE = 256 # Ensure this is consistent with model and preprocessing
IN_CHANNELS = 1
BATCH_SIZE = 64 # Defined earlier
epochs = 50 # Example, can be configured
WARMUP_EPOCHS = 5

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print(f"🌍 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    # print("🌍 Using CPU.")

# %%
def format_time(seconds):
    if seconds < 0: seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"

# %% [markdown]
# ---
# # 🩺 Fine-Tuning Pretrained Architectures ...
# (Content removed for brevity)
# ---

# %%
SAVING_PATH = Path("./pneumonia_efficientnetB0_run") # Example save path
SAVING_PATH.mkdir(parents=True, exist_ok=True)

# %%
clear_memory() # Clear memory before creating new model and optimizer

# %%
# ---------------------------------------------------------------------------- #
#                         💡 MODEL, OPTIMIZER, CRITERION                        #
# ---------------------------------------------------------------------------- #
model_instance = PneumoniaModelCAM(
    pretrained_model_name="efficientnet_b0", # timm handles .ra_in1k etc.
    num_classes=1, in_channels=IN_CHANNELS, criterion_pos_weight=1.0,
    classifier_hidden_dim=512, classifier_dropout_rate=0.3, img_size=IMG_SIZE,
)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # print(f"🚀 Using nn.DataParallel for {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model_instance)
else:
    model = model_instance
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion_obj = model.module.criterion if isinstance(model, nn.DataParallel) else model.criterion
criterion_obj.to(device)
scheduler = None # Example: CosineAnnealingLR(optimizer, T_max=epochs)

train_accuracy = torchmetrics.Accuracy(task="binary").to(device)
train_auroc = torchmetrics.AUROC(task="binary").to(device)
val_accuracy = torchmetrics.Accuracy(task="binary").to(device)
val_auroc = torchmetrics.AUROC(task="binary").to(device)

writer = SummaryWriter(log_dir=str(SAVING_PATH / "logs"))
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

try:
    _model_for_summary = model_instance
    # print(summary(
    #     _model_for_summary,
    #     input_size=(BATCH_SIZE, _model_for_summary.in_channels, _model_for_summary.img_size, _model_for_summary.img_size),
    #     col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=0
    # ))
except Exception as e:
    print(f"⚠️ Model summary failed: {e}")


best_val_auc = 0.0
early_stopping_patience = 10
epochs_no_improve = 0
best_val_loss_for_early_stopping = float("inf")
epoch_durations = []
overall_start_time = time.time()

# print(f"
🚀 Starting training for {epochs} epochs on {device}.")
# ---------------------------------------------------------------------------- #
#                               🏁 TRAINING LOOP 🏁                             #
# ---------------------------------------------------------------------------- #
# Check if train_loader and val_loader are valid before starting loop
if not hasattr(train_loader, 'dataset') or not train_loader.dataset or len(train_loader.dataset) == 0:
    print("❌ Training cannot start: train_loader is empty or invalid.")
elif not hasattr(val_loader, 'dataset') or not val_loader.dataset or len(val_loader.dataset) == 0:
    print("❌ Training cannot start: val_loader is empty or invalid.")
else:
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_accuracy.reset(); train_auroc.reset()
        train_progress_bar = tqdm_auto(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_idx, (img, label) in train_progress_bar:
            img, label = img.to(device), label.float().to(device) # Ensure label is float for BCE
            # Ensure label is [B] or [B,1] for binary classification
            if label.ndim == 1: label = label.unsqueeze(1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_logits, _ = model(img) # pred_logits is [B] if num_classes=1 and squeezed in model
                # Ensure pred_logits is [B,1] if label is [B,1] for criterion
                if pred_logits.ndim == 1: pred_logits = pred_logits.unsqueeze(1)
                loss = criterion_obj(pred_logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item() * img.size(0)
            preds_proba = torch.sigmoid(pred_logits.detach())
            train_accuracy.update(preds_proba, label.int())
            train_auroc.update(preds_proba, label.int())
            if batch_idx % 100 == 0: writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(train_loader) + batch_idx)
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = train_accuracy.compute()
        epoch_train_auc = train_auroc.compute()
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_train_acc, epoch)
        writer.add_scalar("AUROC/train_epoch", epoch_train_auc, epoch)

        model.eval()
        running_val_loss = 0.0
        val_accuracy.reset(); val_auroc.reset()
        val_progress_bar = tqdm_auto(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False)
        with torch.no_grad():
            for img_batch, label_batch in val_progress_bar:
                img_batch, label_batch = img_batch.to(device), label_batch.float().to(device)
                if label_batch.ndim == 1: label_batch = label_batch.unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred_logits, _ = model(img_batch)
                    if pred_logits.ndim == 1: pred_logits = pred_logits.unsqueeze(1)
                    loss = criterion_obj(pred_logits, label_batch)
                running_val_loss += loss.item() * img_batch.size(0)
                preds_proba = torch.sigmoid(pred_logits.detach())
                val_accuracy.update(preds_proba, label_batch.int())
                val_auroc.update(preds_proba, label_batch.int())
                val_progress_bar.set_postfix(loss=loss.item())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute()
        epoch_val_auc = val_auroc.compute()
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val_epoch", epoch_val_acc, epoch)
        writer.add_scalar("AUROC/val_epoch", epoch_val_auc, epoch)

        duration_epoch_seconds = time.time() - epoch_start_time
        epoch_durations.append(duration_epoch_seconds)
        # print(f"Epoch {epoch+1} Done. Val AUC: {epoch_val_auc:.4f}")

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            # Save checkpoint
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'val_loss': avg_val_loss
            }
            model_save_name = SAVING_PATH / f"best_model_epoch_{epoch+1}_auc_{best_val_auc:.4f}.pth"
            torch.save(checkpoint_data, model_save_name)
            # print(f"🏅 Checkpoint saved: {model_save_name}")


        if avg_val_loss < best_val_loss_for_early_stopping:
            best_val_loss_for_early_stopping = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            # print(f"✋ Early stopping at epoch {epoch+1}.")
            break
        if scheduler: scheduler.step(avg_val_loss if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None)
    writer.close()
    # print("
--- 🏆 Training Finished ---")


# %%
# Path to a pre-trained model checkpoint (replace with your actual path if needed)
# This is an example path, it might not exist in the environment.
checkpoint_path_example = SAVING_PATH / "best_model_epoch_49_auc_0.9985.pth" # Example

if checkpoint_path_example.exists():
    # print(f"Loading model from {checkpoint_path_example}")
    checkpoint = torch.load(checkpoint_path_example, map_location=device)

    # Re-initialize model instance BEFORE loading state_dict
    # Ensure parameters match the saved model
    model_loaded = PneumoniaModelCAM(
        pretrained_model_name="efficientnet_b0", # Or the one used for the saved model
        num_classes=1, in_channels=IN_CHANNELS, img_size=IMG_SIZE
        # Ensure other params like classifier_hidden_dim match if they affect architecture
    )
    # Handle DataParallel state_dict if saved from DataParallel
    state_dict = checkpoint['model_state_dict']
    if isinstance(model, nn.DataParallel): # If current model is DP
        model_loaded.load_state_dict(state_dict) # Assumes saved state_dict is not from DP
    else: # If current model is not DP
        # Check if state_dict needs stripping of 'module.' prefix
        if all(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model_loaded.load_state_dict(new_state_dict)
        else:
            model_loaded.load_state_dict(state_dict)

    model_loaded.to(device)
    model_loaded.eval()
    # print("Model loaded successfully and set to eval mode.")
else:
    # print(f"Checkpoint not found at {checkpoint_path_example}. Using the model from training (if any).")
    model_loaded = model # Fallback to model from training loop if checkpoint not found
    model_loaded.eval()


# %% [markdown]
# ---
# ## 🎨 Visualizing Your Vanilla PyTorch `PneumoniaModelCAM` with Netron ...
# (Content removed for brevity)
# ---

# %%
# ONNX Export
# Ensure model_loaded is defined (either from checkpoint or end of training)
if 'model_loaded' in globals() and isinstance(model_loaded, nn.Module):
    dummy_input_for_export = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
    # If model is on GPU, dummy input should be too.
    # If model is .half(), dummy_input might need to be .half() too.
    if next(model_loaded.parameters()).is_cuda:
        dummy_input_for_export = dummy_input_for_export.cuda()
    if next(model_loaded.parameters()).dtype == torch.float16:
        dummy_input_for_export = dummy_input_for_export.half()

    onnx_model_path = SAVING_PATH / "pneumonia_model.onnx"
    try:
        torch.onnx.export(
            model_loaded, dummy_input_for_export, onnx_model_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['input_image'], output_names=['logits', 'features_map'],
            dynamic_axes={
                'input_image' : {0 : 'batch_size'},
                'logits' : {0 : 'batch_size'},
                'features_map' : {0 : 'batch_size'}
            }
        )
        # print(f"✅ Model exported to ONNX: {onnx_model_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
else:
    print("⚠️ Model for ONNX export ('model_loaded') not defined.")


# %%
# CAM Generation Function (modified for PneumoniaModelCAM)
def generate_cam_for_pneumonia_model(
    model_to_eval: PneumoniaModelCAM, # Expects the actual PneumoniaModelCAM instance
    img_tensor: torch.Tensor,
    target_size: Tuple[int, int] = (256, 256) # Original image size for overlay
) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:

    model_to_eval.eval()
    model_device = next(model_to_eval.parameters()).device
    model_dtype = next(model_to_eval.parameters()).dtype

    processed_img_tensor = img_tensor.to(model_device, dtype=model_dtype)
    if processed_img_tensor.ndim == 2: processed_img_tensor = processed_img_tensor.unsqueeze(0) # C
    if processed_img_tensor.ndim == 3: processed_img_tensor = processed_img_tensor.unsqueeze(0) # B

    with torch.no_grad():
        logits, feature_maps = model_to_eval(processed_img_tensor) # feature_maps: [B, C_feat, H_feat, W_feat]

    # Use the EnhancedClassifier's final linear layer weights for CAM
    # This is a common approach for CAM with a custom classifier.
    # The weights connect the pooled features to the class scores.
    try:
        # Access the last linear layer of the EnhancedClassifier
        # The EnhancedClassifier is 'classifier' attribute of PneumoniaModelCAM
        # The Sequential is 'feature_map_processor'
        # The last nn.Linear is at index -1 in the Sequential
        final_classifier_layer = model_to_eval.classifier.feature_map_processor[-1]
        if not isinstance(final_classifier_layer, nn.Linear):
            # print("Error: Last layer of classifier is not nn.Linear. CAM might be incorrect.")
            return None, logits.squeeze().cpu()

        # Weights shape: [num_classes, num_backbone_out_features] if num_classes > 1
        # or [1, num_backbone_out_features] if num_classes = 1
        # For binary, we take the weights for the positive class (index 0 if num_classes=1)
        classifier_weights = final_classifier_layer.weight.squeeze() # Shape [num_backbone_out_features]

        # CAM calculation: feature_maps [B, C_feat, H_feat, W_feat], weights [C_feat]
        # We need to ensure C_feat matches the dimension of classifier_weights
        if feature_maps.shape[1] != classifier_weights.shape[0]:
            # print(f"Error: Mismatch in feature map channels ({feature_maps.shape[1]}) and classifier weights ({classifier_weights.shape[0]})")
            return None, logits.squeeze().cpu()

        # Weighted sum of feature maps
        # cam = torch.einsum('cfhw,c->fhw', feature_maps.squeeze(0), classifier_weights) # if batch_size=1
        cam = torch.zeros(feature_maps.shape[0], feature_maps.shape[2], feature_maps.shape[3], device=model_device, dtype=model_dtype)
        for i in range(feature_maps.shape[0]): # Iterate over batch if B > 1
             for c_idx in range(feature_maps.shape[1]):
                 cam[i] += feature_maps[i, c_idx, :, :] * classifier_weights[c_idx]


    except Exception as e:
        # print(f"Error during CAM weight extraction or calculation: {e}")
        return None, logits.squeeze().cpu()


    # Normalize CAM: 0 to 1
    cam_normalized = cam.squeeze().detach().cpu().numpy() # Ensure it's 2D if B=1
    if cam_normalized.ndim > 2 and cam_normalized.shape[0] == 1: # Handle if batch was > 1 and we only want first
        cam_normalized = cam_normalized[0]

    if np.max(cam_normalized) - np.min(cam_normalized) > 1e-6 :
        cam_normalized = (cam_normalized - np.min(cam_normalized)) / (np.max(cam_normalized) - np.min(cam_normalized))
    else:
        cam_normalized = np.zeros_like(cam_normalized)


    # Resize CAM to original image size for overlay
    cam_resized = cv2.resize(cam_normalized, target_size, interpolation=cv2.INTER_LINEAR)
    return cam_resized, logits.squeeze().cpu()


# %%
def visualize_cam_overlay(
    img_tensor_orig: torch.Tensor, # Original image tensor (before model processing)
    cam_map: np.ndarray,
    model_prediction_logits: torch.Tensor,
    is_positive_threshold: float = 0.0, # Logits space threshold
):
    img_display = img_tensor_orig.detach().cpu()
    if img_display.ndim == 4 and img_display.shape[0] == 1: img_display = img_display.squeeze(0) # B,C,H,W -> C,H,W
    if img_display.ndim == 3 and img_display.shape[0] == 1: img_display = img_display.squeeze(0) # C,H,W -> H,W (if grayscale)
    elif img_display.ndim == 3 and img_display.shape[0] == 3: img_display = img_display.permute(1,2,0) # C,H,W -> H,W,C (if RGB)

    img_np = img_display.numpy().astype(np.float32)
    if img_np.max() > 1.0 or img_np.min() < 0.0: # Normalize if not already
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    if img_np.ndim == 3 and img_np.shape[-1] !=3 and img_np.shape[-1] ==1: # Handle (H,W,1)
        img_np = img_np.squeeze(-1)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap="bone")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_np, cmap="bone")
    plt.imshow(cam_map, alpha=0.5, cmap="jet") # Overlay CAM
    prob = torch.sigmoid(model_prediction_logits).item()
    is_pos = model_prediction_logits.item() > is_positive_threshold
    plt.title(f"Prediction Positive: {is_pos} (Prob: {prob:.3f}, Logit: {model_prediction_logits.item():.3f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# Example of using CAM visualization
if SAVE_PATH.exists():
    # Try to pick an image from the validation set used during training
    # This assumes 'val_dataset' is still defined and populated
    if 'val_dataset' in globals() and hasattr(val_dataset, 'samples') and val_dataset.samples:
        random_idx = random.randint(0, len(val_dataset)-1)
        original_img_tensor, label = val_dataset[random_idx] # This image is already transformed by val_transforms
        # For visualization, we might want the *untransformed* or less transformed image if possible
        # Or, we can de-normalize 'original_img_tensor' if needed.
        # For now, we use the transformed one.
        # print(f"Visualizing CAM for a sample from val_dataset (index {random_idx}). Label: {label.item()}")
    else: # Fallback to loading a random .npy file if val_dataset is not available
        # print("val_dataset not available, loading a random .npy file for CAM.")
        split_dir_for_cam = SAVE_PATH / "val" / "0" # Example: healthy class from validation
        if not split_dir_for_cam.exists(): split_dir_for_cam = SAVE_PATH / "train" / "0" # Fallback to train
        if split_dir_for_cam.exists():
            npy_paths_for_cam = list(split_dir_for_cam.rglob("*.npy"))
            if npy_paths_for_cam:
                random_npy_path_for_cam = random.choice(npy_paths_for_cam)
                npy_file_for_cam = np.load(random_npy_path_for_cam).astype(np.float32)
                # This loaded .npy is typically (H,W) and normalized.
                # We need to apply ToTensor but not the normalization again if it's already normalized.
                original_img_tensor = torch.from_numpy(npy_file_for_cam) # Shape [H,W]
                # print(f"Visualizing CAM for {random_npy_path_for_cam}")
            else:
                original_img_tensor = None
                # print(f"No .npy files found in {split_dir_for_cam}")
        else:
            original_img_tensor = None
            # print(f"Directory {split_dir_for_cam} not found for CAM example.")

    if original_img_tensor is not None and 'model_loaded' in globals():
        # Ensure model_loaded is PneumoniaModelCAM, not DataParallel wrapper for CAM generation
        model_for_cam = model_loaded.module if isinstance(model_loaded, nn.DataParallel) else model_loaded

        cam_generated, logits_generated = generate_cam_for_pneumonia_model(
            model_for_cam,
            original_img_tensor, # This should be the tensor input to the model
            target_size=(original_img_tensor.shape[-2], original_img_tensor.shape[-1]) # Use actual H,W
        )
        if cam_generated is not None:
            visualize_cam_overlay(original_img_tensor, cam_generated, logits_generated)
        else:
            print("CAM generation failed.")
    elif original_img_tensor is None:
        print("No image tensor available for CAM visualization.")
    elif 'model_loaded' not in globals():
        print("Model 'model_loaded' not defined for CAM visualization.")

else:
    print(f"SAVE_PATH {SAVE_PATH} does not exist. Cannot load image for CAM.")
