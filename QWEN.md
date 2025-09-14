# ECI CAPTCHA Solver - Qwen Context

## Project Overview

This is a production-ready CAPTCHA solver for the Election Commission of India (ECI) portal, specifically designed to handle 200x80 pixel CAPTCHA images. The system uses advanced preprocessing techniques, machine learning models (CRNN and segmentation-based), and GPU acceleration to achieve high accuracy in solving ECI CAPTCHAs.

## Key Components

### Core Solver
- `main_solver.py` - The primary CAPTCHA solver with CRNN and segmentation models

### Preprocessing
- `preprocessing_pixel.py` - Pixel-level preprocessing with precise mathematical approaches

### GPU Acceleration
- `gpu_accelerator.py` - GPU acceleration manager using PyTorch CUDA or TensorFlow

### Model Training and Management
- `train_all_models.py` - Script to train models with all preprocessing methods
- `test_model_loading.py` - Verification of model loading functionality
- `MODELS_README.md` - Documentation on multi-model support

### Testing Framework
- `test_framework.py` - Comprehensive testing suite
- `test_crnn_model.py` - CRNN model output shape verification
- `test_crnn_training.py` - CRNN model training verification

### Environment Setup
- `setup_environment.py` - Complete environment setup with dependencies
- `setup_directml_env.py` - DirectML environment setup for Windows

## Technology Stack

- **Python 3.11** - Primary programming language
- **TensorFlow 2.20** - Deep learning framework for CRNN model
- **PyTorch** - Alternative deep learning framework for GPU acceleration
- **OpenCV** - Image processing and computer vision
- **NumPy/Pandas** - Numerical computing and data handling
- **Scikit-learn** - Machine learning utilities
- **CUDA** - GPU acceleration for NVIDIA GPUs

## Project Structure

```
eci_captcha_solver/
├── captcha_dataset/     # Training data (PNG + JSON files)
├── models/             # Trained models
├── config/             # Configuration files
├── scripts/            # Utility scripts
├── logs/               # Training logs
├── output/             # Test results
├── main_solver.py      # Core solver
├── setup_environment.py # Environment setup
└── test_framework.py   # Testing suite
```

## Multi-Model Support

The system supports training and using separate models for each preprocessing method:
1. `grayscale` - Basic preprocessing that converts images to grayscale
2. `light_smoothing` - Applies light smoothing to reduce noise while preserving edges
3. `precise_mathematical` - Uses precise mathematical approach for line removal

Models are saved with method-specific suffixes and automatically loaded based on the selected preprocessing method.

## GPU Acceleration

The system includes GPU acceleration support using:
- PyTorch CUDA (primary for Windows)
- TensorFlow GPU (fallback)
- CPU-only mode (when GPU not available)

GPU acceleration is used for both training and inference to improve performance.

## Development Conventions

### Code Style
- Python 3.11 with type hints
- Object-oriented design with clear class separation
- Comprehensive logging for debugging and monitoring
- Error handling with descriptive messages

### Model Architecture
- CRNN (Convolutional Recurrent Neural Network) for sequence recognition
- Segmentation-based model for individual character recognition
- Ensemble approach combining multiple models for better accuracy

### Preprocessing Pipeline
- Text-preserving noise removal
- Line detection and removal
- Image enhancement techniques
- Multiple preprocessing methods for different CAPTCHA variations

## Key Features

- Optimized for 200x80 ECI images
- Text-preserving noise removal
- Ensemble model approach
- GPU acceleration
- Continuous learning
- Production monitoring
- Multi-model support for different preprocessing methods

## Building and Running

### Environment Setup
```bash
python setup_environment.py --install-deps --create-structure
```

### Training
```bash
# Train models with all preprocessing methods
python train_all_models.py captcha_dataset

# Train model with specific preprocessing method
python main_solver.py --mode train --dataset-dir captcha_dataset --preprocessing-method light_smoothing
```

### Testing
```bash
python main_solver.py --mode test --dataset-dir captcha_dataset
```

### Solving CAPTCHAs
```bash
# Solve single image
python main_solver.py --mode solve --image-path test_captcha.png

# Solve with specific preprocessing method
python main_solver.py --mode solve --image-path captcha_dataset --preprocessing-method precise_mathematical
```

### Integration
```python
# Test single image
from scripts.captcha_integration import solve_captcha_image
solution = solve_captcha_image("captcha.png")
```

## Recent Fixes

### CRNN Model Shape Mismatch Issue
A recent fix addressed a shape mismatch issue in the CRNN model where the model was outputting tensors with incorrect dimensions. The fix involved:
1. Modifying the model's output layer to correctly map the RNN output to the expected character sequence shape
2. Splitting the 25 time steps into 6 groups and averaging each group to get predictions for each character position
3. Ensuring the model's output shape matches the expected shape for the loss function (batch_size, 6, 31)
