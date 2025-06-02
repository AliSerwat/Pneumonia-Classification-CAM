import random
import numpy as np
import torch
import gc
from pathlib import Path
from typing import Union, Optional, List # Added Optional, List

def set_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility across random number generators.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: For full reproducibility, you might also consider:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed} for random, numpy, and torch.")

def get_device() -> torch.device:
    """
    Determines and returns the appropriate torch.device, printing its information.

    Returns:
        torch.device: The determined torch device (cuda or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🌍 Using GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"Found {torch.cuda.device_count()} GPUs. Note: DataParallel or DistributedDataParallel needs to be handled by the training script if multiple GPUs are to be used effectively.")
    else:
        device = torch.device("cpu")
        print("🌍 Using CPU.")
    return device

def format_time(seconds: float) -> str:
    """
    Formats time in seconds to a human-readable string HHh:MMm:SSs.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: The formatted time string.
    """
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"

def clear_memory() -> None:
    """
    Performs garbage collection and empties PyTorch CUDA cache.
    Note: Global variable deletion is best handled in the scope where those variables exist.
    """
    print("🧹 Running garbage collection...")
    gc.collect()
    if torch.cuda.is_available():
        print("🅿️ Emptying PyTorch CUDA cache...")
        torch.cuda.empty_cache()  # Clears unused memory from PyTorch's cache
        # Optional: torch.cuda.synchronize() might be useful if you want to ensure
        # all CUDA operations are complete before proceeding, though empty_cache itself
        # is generally synchronous with respect to freeing memory.
        # Forcing synchronization can sometimes help in specific debugging scenarios.
        # torch.cuda.synchronize()
        print("✅ PyTorch CUDA cache emptied.")
    else:
        print("ℹ️ CUDA not available. Skipping CUDA cache clear.")
    print("✅ Memory clearing actions complete.")

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
        # Check if the path exists and is a directory
        if not path.exists():
            # print(f"Directory {path} does not exist. Considering it empty.")
            return True
        if not path.is_dir():
            # print(f"Path {path} is not a directory.")
            return False # Or raise an error, depending on desired behavior
        # Check if any items are in the directory
        return not any(path.iterdir())
    except FileNotFoundError: # Should be caught by path.exists() but as a fallback
        # print(f"FileNotFoundError for {path}. Considering it empty.")
        return True
    except Exception as e: # Catch other potential errors like permission issues
        print(f"Error checking directory {path}: {e}")
        return False # Or re-raise, depending on how strict it should be
