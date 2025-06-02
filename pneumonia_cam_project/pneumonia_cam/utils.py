import os
import gc
import math
import json
import time
import shutil
import random
import logging
import subprocess
from pathlib import Path
from typing import Union, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd

# ============================ ⚙️ Basic Configuration ============================ #

def get_device(verbose: bool = True) -> torch.device:
    """
    Determines and returns the appropriate PyTorch device (GPU or CPU).

    Args:
        verbose (bool): If True, prints information about the selected device.

    Returns:
        torch.device: The selected PyTorch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"🌍 Using GPU: {torch.cuda.get_device_name(0)}")
            if torch.cuda.device_count() > 1:
                print(
                    f"Found {torch.cuda.device_count()} GPUs. "
                    "Consider using DataParallel or DistributedDataParallel."
                )
    else:
        device = torch.device("cpu")
        if verbose:
            print("🌍 Using CPU. Operations may be slower.")
    return device

DEVICE = get_device()

def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # The following two lines are often recommended for reproducibility,
        # but can impact performance. Enable if strict reproducibility is paramount.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to {seed}")

# ============================ 🕒 Time Utilities ============================ #

def format_time(seconds: float) -> str:
    """
    Formats time in seconds to a human-readable string HHh:MMm:SSs.

    Args:
        seconds (float): The total number of seconds.

    Returns:
        str: The formatted time string.
    """
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"

# ============================ 🧹 Memory Utilities ============================ #

def clear_memory(vars_to_delete: Optional[List[str]] = None) -> None:
    """
    Attempts to clear specified global variables and PyTorch CUDA cache.

    This function performs several steps to free up memory:
    1. Deletes specified global variables from the caller's global scope.
    2. Runs Python's garbage collector.
    3. Empties the PyTorch CUDA memory cache if CUDA is available.
    4. Synchronizes CUDA operations to ensure completion.
    5. Reports the current GPU memory usage if CUDA is available.

    Args:
        vars_to_delete (Optional[List[str]]): A list of names of global variables
                                                to delete. Defaults to a predefined list
                                                of common large objects if None.
                                                Note: This function can only delete
                                                variables from the scope where it's called
                                                if `globals()` is passed correctly or
                                                handled carefully. For simplicity, this
                                                implementation assumes it's called from
                                                a context where `globals()` refers to the
                                                desired scope or that `vars_to_delete`
                                                are names of variables in the global scope
                                                of this utils module if not handled by caller.
                                                A more robust way would be to pass the dictionary
                                                from `globals()` of the calling module.
    """
    # This function's ability to delete vars from the *caller's* scope is limited
    # when called from another module. The `globals()` here refers to `utils.py`'s globals.
    # For effective cleanup in a script, this logic might need to be in the script itself
    # or `vars_to_delete` should be actual objects passed for deletion.

    print("🧹 Attempting to clear memory (Note: variable deletion effectiveness depends on calling context)...")

    if vars_to_delete is None:
        vars_to_delete_names = [
            "inputs", "model", "processor", "trainer", "peft_model", "bnb_config",
            "optimizer", "scheduler", "train_dataset", "eval_dataset", "data_collator",
        ]
    else:
        vars_to_delete_names = vars_to_delete

    # This part will only work if these variables are global in this module's scope
    # or if the caller explicitly passes them.
    # For now, it serves as a template.
    # deleted_vars_count = 0
    # for var_name in vars_to_delete_names:
    #     if var_name in globals(): # Checks globals() of utils.py
    #         del globals()[var_name]
    #         deleted_vars_count += 1
    #         print(f"   - Deleted '{var_name}' from utils.py global scope (if it existed there)")

    # print(f"ℹ️ Note: Variable deletion from caller's scope is complex. Focus is on GC and CUDA cache.")

    time.sleep(0.1) # Short pause

    print("\n♻️ Running Python's garbage collector...")
    gc.collect()
    print("✅ Garbage collection complete.")

    time.sleep(0.1)

    if torch.cuda.is_available():
        print("\n🅿️ Managing PyTorch CUDA memory...")
        torch.cuda.empty_cache()
        print("   - Emptied PyTorch CUDA cache.")
        torch.cuda.synchronize()
        print("   - Synchronized CUDA operations.")
        print("✅ PyTorch CUDA memory management actions complete.")

        print("\n📊 GPU Memory Status After Cleanup:")
        allocated_mem = torch.cuda.memory_allocated() / (1024**3)
        reserved_mem = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU allocated memory: {allocated_mem:.2f} GB")
        print(f"   GPU reserved memory : {reserved_mem:.2f} GB")
    else:
        print("\nℹ️ CUDA not available. Skipping PyTorch CUDA memory management.")

    time.sleep(0.1)
    print("\n♻️ Running a final garbage collection pass...")
    gc.collect()
    print("✅ Final garbage collection complete.")

# ============================ 📁 File & Directory Utilities ============================ #

def is_directory_empty(directory_path: Union[str, Path]) -> bool:
    """
    Checks if a directory is empty.

    Args:
        directory_path (Union[str, Path]): The path to the directory.

    Returns:
        bool: True if the directory is empty or does not exist, False otherwise.
    """
    path = Path(directory_path)
    try:
        return not any(path.iterdir())
    except FileNotFoundError:
        return True  # Consider it empty if the directory doesn't exist

def setup_kaggle_credentials(username: str, key: str) -> None:
    """
    Sets up Kaggle API credentials.

    Args:
        username (str): Kaggle username.
        key (str): Kaggle API key.
    """
    home_dir = Path.home()
    kaggle_config_dir = home_dir / ".kaggle"
    os.makedirs(kaggle_config_dir, exist_ok=True)

    kaggle_json_path = kaggle_config_dir / "kaggle.json"
    kaggle_credentials = {"username": username, "key": key}

    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_credentials, f)

    os.chmod(kaggle_json_path, 0o600)
    print(f"Kaggle API key saved to: {kaggle_json_path}")

def download_and_extract_dataset(
    dataset_name: str, download_path: Path, extract_path: Path
) -> None:
    """
    Downloads a Kaggle dataset and extracts it.

    Args:
        dataset_name (str): The name of the Kaggle dataset (e.g., "rsna-pneumonia-detection-challenge").
        download_path (Path): The directory where the dataset zip file will be downloaded.
        extract_path (Path): The directory where the dataset will be extracted.
    """
    zip_file_path = download_path / f"{dataset_name}.zip"

    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    if zip_file_path.exists():
        print(f"Dataset '{dataset_name}.zip' already exists at '{download_path}'.")
    else:
        try:
            print(f"Downloading '{dataset_name}' to '{download_path}'...")
            subprocess.run(
                [
                    "kaggle", "competitions", "download", "-c", dataset_name,
                    "-p", str(download_path),
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print("Download complete.")
        except FileNotFoundError:
            print("Error: Kaggle CLI not found. Please install it (pip install kaggle) and configure credentials.")
            return
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return

    if zip_file_path.exists():
        if is_directory_empty(extract_path):
            print(f"Extracting '{zip_file_path.name}' to '{extract_path}'...")
            try:
                subprocess.run(
                    ["unzip", "-q", str(zip_file_path), "-d", str(extract_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("Extraction complete.")
            except FileNotFoundError:
                print("Error: 'unzip' command not found. Please ensure it's installed.")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting dataset: {e}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
        else:
            print(f"Warning: Extract directory '{extract_path}' is not empty. Skipping extraction.")
    else:
        print(f"Error: Zip file '{zip_file_path.name}' not found after download attempt.")

if __name__ == "__main__":
    # Example usage of some utility functions
    print("Device:", DEVICE)
    seed_everything(123)
    print("Formatted time for 3661 seconds:", format_time(3661))

    # Create dummy directories/files for testing is_directory_empty
    test_empty_dir = Path("./test_empty_dir_utils")
    test_non_empty_dir = Path("./test_non_empty_dir_utils")

    os.makedirs(test_empty_dir, exist_ok=True)
    os.makedirs(test_non_empty_dir, exist_ok=True)
    with open(test_non_empty_dir / "dummy.txt", "w") as f:
        f.write("hello")

    print(f"Is '{test_empty_dir}' empty? {is_directory_empty(test_empty_dir)}")
    print(f"Is '{test_non_empty_dir}' empty? {is_directory_empty(test_non_empty_dir)}")
    print(f"Is './non_existent_dir' empty? {is_directory_empty('./non_existent_dir_utils')}")

    # Clean up dummy directories
    shutil.rmtree(test_empty_dir, ignore_errors=True)
    shutil.rmtree(test_non_empty_dir, ignore_errors=True)

    # clear_memory() # Example, but be careful with global var deletion

    # Example for dataset download (requires Kaggle API setup)
    # print("\n--- Example Dataset Download (commented out by default) ---")
    # KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME") # Or set directly
    # KAGGLE_KEY = os.environ.get("KAGGLE_KEY")           # Or set directly
    # if KAGGLE_USERNAME and KAGGLE_KEY:
    #     setup_kaggle_credentials(KAGGLE_USERNAME, KAGGLE_KEY)
    #     current_project_dir = Path.cwd().parent # Assuming this script is in pneumonia_cam
    #     data_dir = current_project_dir / "data" # Example data directory
    #     rsna_dataset_raw_path = data_dir / "raw" / "rsna-pneumonia-detection-challenge"
    #     download_and_extract_dataset(
    #         "rsna-pneumonia-detection-challenge",
    #         download_path=data_dir / "raw",
    #         extract_path=rsna_dataset_raw_path
    #     )
    # else:
    #     print("Kaggle username/key not found in environment variables. Skipping dataset download example.")
