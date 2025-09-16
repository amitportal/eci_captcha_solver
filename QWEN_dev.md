# ECI CAPTCHA Solver - Qwen Context

## Project Overview

This is a production-ready CAPTCHA solver for the Election Commission of India (ECI) portal, specifically designed to handle 200x80 pixel CAPTCHA images. The system uses advanced preprocessing techniques, machine learning models (CRNN and segmentation-based), and GPU acceleration to achieve high accuracy in solving ECI CAPTCHAs.

## Key Components

### Core Solver
- `main.py` - The primary CAPTCHA solver with CRNN and segmentation models
- `captcha_solver.py` - Fast independent CAPTCHA solver with minimal preprocessing

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
├── main.py      # Core solver
├── captcha_solver.py   # Fast independent solver
├── setup_environment.py # Environment setup
└── test_framework.py   # Testing suite
```

## Multi-Model Support

The system supports training and using separate models for each preprocessing method:
1. `grayscale` - Basic preprocessing that converts images to grayscale
2. `light_smoothing` - Applies light smoothing to reduce noise while preserving edges
3. `precise_mathematical` - Uses precise mathematical approach for line removal
4. `raw` - Raw preprocessing that only resizes and normalizes images

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
- CRNN (Convolutional Recurrent Neural Network) using SimpleCRNN architecture for sequence recognition
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
- Fast independent solver (`captcha_solver.py`)
- Batch processing capabilities

## Fast Independent Solver (captcha_solver.py)

The `captcha_solver.py` script is an optimized, independent solver designed for maximum speed:
- **Minimal preprocessing**: Only converts to grayscale and normalizes (no image resizing or complex operations)
- **GPU acceleration**: Automatically uses PyTorch CUDA when available for faster model loading
- **Model caching**: Caches the model in memory during execution for faster subsequent solves (within single execution)
- **Batch processing**: Can solve multiple images in a single execution for improved performance
- **Clean output**: Prints only the solution to stdout, with detailed timing in log file
- **Performance**: ~0.7 seconds total solve time (0.2s model load + 0.5s prediction) with GPU

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
python main.py --mode train --dataset-dir captcha_dataset --preprocessing-method light_smoothing
```

### Testing
```bash
python main.py --mode test --dataset-dir captcha_dataset
```

### Solving CAPTCHAs
```bash
# Solve single image
python main.py --mode solve --image-path test_captcha.png

# Solve with specific preprocessing method
python main.py --mode solve --image-path captcha_dataset --preprocessing-method precise_mathematical

# Solve with CRNN model (raw)
    # Solve a single image
    python captcha_solver.py test_captcha.png
    
    # Solve multiple images
    python captcha_solver.py test_captcha_3.png test_captcha_4.png
    
    # Solve multiple images with paths containing spaces (use quotes)
    python captcha_solver.py "path/to/image 1.png" "path/to/image 2.png"
    
    # Solve all PNG images in current directory
    python captcha_solver.py *.png
    
    # Solve all PNG images in another directory
    python captcha_solver.py other_captchas/*.png
    
    # Solve all PNG images in a directory
    python captcha_solver.py test_images/
    
    # Or, for files with a prefix:
    python captcha_solver.py other_captchas_*.png
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

### Simplified CRNN Architecture
The CRNN model architecture has been simplified to use only the SimpleCRNN implementation:
1. Removed the complex OptimizedCRNN class
2. All CRNN models now use the simpler, more stable SimpleCRNN architecture
3. This change applies to all preprocessing methods (grayscale, light_smoothing, precise_mathematical, raw)
4. Models are saved as `simple_crnn_200x80_{preprocessing_method}.keras`

### Training Improvements
Recent improvements to the training process include:
1. **Extended Training Duration**: Increased training epochs from 50 to 100 for better model convergence
2. **Improved Early Stopping**: Increased patience from 5-15 epochs to 20-25 epochs to prevent premature stopping
3. **Better Learning Rate Control**: Extended patience for learning rate reduction from 3-7 epochs to 10 epochs
4. **Enhanced Metrics Collection**: All training metrics files now include preprocessing method suffixes for better tracking
5. **Clear Section Headers**: Added better demarcation in logs between segmentation and CRNN training phases
6. **Method-Specific Model Naming**: All models are saved with preprocessing method suffixes for easy identification

### Fast Solver Optimizations
Recent improvements to the fast CAPTCHA solver (`captcha_solver.py`) include:
1. **GPU Acceleration**: Now automatically uses PyTorch CUDA when available, reducing model loading time from ~0.5s to ~0.2s
2. **Minimal Preprocessing**: Removed all unnecessary image operations, focusing only on essential conversions
3. **Detailed Logging**: Added comprehensive timing information to log file without cluttering terminal output
4. **Model Caching**: Implements caching within single execution to avoid reloading model multiple times
5. **Clean Output**: Terminal output contains only the solution, with detailed timing in log file
6. **Batch Processing**: Can solve multiple images in a single execution with improved performance for subsequent images

## Qwen Added Memories
- "check all files in the working directory, don't ignore files categorised by @.gitignore"
