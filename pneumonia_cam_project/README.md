# Pneumonia Detection with Class Activation Mapping (CAM)

This project implements a deep learning model for detecting pneumonia in chest X-ray images. It leverages Class Activation Mapping (CAM) to visualize the regions of an image that the model uses to make its predictions, providing interpretability alongside diagnostic aid.

The codebase is refactored from an exploratory Jupyter Notebook into a structured Python package suitable for further development and deployment.

## Project Overview

The core components of this project include:

*   **Data Loading & Preprocessing**: Handles loading of DICOM images from datasets like the RSNA Pneumonia Detection Challenge, preprocessing them into a usable format (e.g., normalized NumPy arrays), and preparing PyTorch Datasets and DataLoaders.
*   **Model Architecture**: Implements `PneumoniaModelCAM`, a convolutional neural network (CNN) that utilizes a pre-trained backbone (e.g., ResNet, EfficientNet via `timm`) augmented with a custom classifier head. The model is designed to output both predictions and feature maps necessary for CAM generation.
*   **Training**: Provides a script to train the model, including features like weighted sampling for class imbalance, learning rate scheduling, mixed-precision training, checkpointing, and early stopping.
*   **Evaluation**: Offers tools to evaluate trained models, compute various performance metrics (Accuracy, AUROC, Precision, Recall, F1-score, Confusion Matrix), and generate/visualize Class Activation Maps.
*   **Utilities**: Includes helper functions for device management, reproducibility, and other common tasks.

## Code Structure

The project is organized into the following main components:

*   `pneumonia_cam_project/`: The root directory for the project.
    *   `pneumonia_cam/`: The main Python package containing the core logic.
        *   `__init__.py`: Makes `pneumonia_cam` a package.
        *   `data_loader.py`: Contains `PneumoniaDataset`, data transformations, preprocessing functions (`normalize_and_save_images`), and data balancing utilities.
        *   `model.py`: Defines the neural network architectures, including `MedicalAttentionGate`, `EnhancedClassifier`, and `PneumoniaModelCAM`.
        *   `train.py`: Script for training the model.
        *   `evaluate.py`: Script for evaluating trained models, generating CAMs, and exporting to ONNX.
        *   `utils.py`: Common utility functions.
    *   `notebooks/`: (Placeholder) For Jupyter notebooks, e.g., the original exploratory notebook or new analysis notebooks.
    *   `assets/`: (Placeholder) For static assets, such as sample images for a web UI.
    *   `experiments/`: (Placeholder, typically created by `train.py`) To store training runs, checkpoints, and logs.
    *   `requirements.txt`: Lists Python package dependencies.
    *   `setup.py`: Script for packaging and distributing the `pneumonia_cam` library.
    *   `README.md`: This file.
    *   `LICENSE`: (To be added) Project license information.
    *   `CONTRIBUTING.md`: (To be added) Guidelines for contributing to the project.

## Installation

(Placeholder - Instructions will be added here)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd pneumonia_cam_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```

## Usage

(Placeholder - Detailed usage examples will be added here)

### Data Preprocessing

If you are starting with raw DICOM images, you may need to preprocess them first. The `train.py` script has an option for this, or you can adapt functions from `data_loader.py`.

### Training a Model

Use the `train.py` script. Example:
```bash
python -m pneumonia_cam.train \
    --project_root_dir . \
    --dataset_csv_path path/to/your/labels.csv \
    --raw_dicom_dir path/to/your/dicom_images \
    --processed_data_dir PreprocessedData \
    --save_dir training_output \
    --model_name efficientnet_b0 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --preprocess_data \
    --use_weighted_sampler \
    --use_amp
```
Run `python -m pneumonia_cam.train --help` for all available options.

### Evaluating a Model

Use the `evaluate.py` script with a trained model checkpoint. Example:
```bash
python -m pneumonia_cam.evaluate \
    --checkpoint_path training_output/your_model_run/best_model_epoch_X.pth \
    --project_root_dir . \
    --processed_data_dir PreprocessedData \
    --eval_split val \
    --generate_cam \
    --num_cam_samples 5
```
Run `python -m pneumonia_cam.evaluate --help` for all available options.

### Web Application UI

(Placeholder - Information about the web UI will be added once implemented.)

## Contributing

(Placeholder - Contribution guidelines will be detailed in `CONTRIBUTING.md`)

## License

(Placeholder - License information will be provided in the `LICENSE` file.)
