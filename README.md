# ECI CAPTCHA Solver - Phase 4 Implementation

## 🎯 Overview
Production-ready CAPTCHA solver for Election Commission of India (ECI) portal with 200x80 pixel images.

## 📁 Directory Structure
```
eci_captcha_solver/
├── captcha_dataset/     # Training data (PNG + JSON files)
├── models/             # Trained models
├── config/             # Configuration files
├── scripts/            # Utility scripts
├── logs/               # Training logs
├── output/             # Test results
├── main.py             # Core trainer, Data manager, Fallback Solver
├── captcha_solver.py   # Fast independent CAPTCHA solver (minimal preprocessing)
├── setup_environment.py # Environment setup
└── test_framework.py   # Testing suite
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
python setup_environment.py --install-deps --create-structure
```

### 2. Dataset Preparation
```bash
# Place your PNG/JSON files in captcha_dataset/
python scripts/data_consolidator.py captcha_dataset
```

### 3. Training
```bash
python main.py --mode train --dataset-dir captcha_dataset --csv-file captcha_solutions_consolidated.csv --save-preprocessed --use-gpu force
```

### 4. Training All Models with Different Preprocessing Methods
```bash
python train_all_models.py captcha_dataset captcha_solutions_consolidated.csv
```

### 5. Testing
```bash
python main.py --mode test --dataset-dir captcha_dataset --csv-file captcha_solutions_consolidated.csv
```

### 6. Solving CAPTCHAs
```bash
# Solve a single image with default settings
python main.py --mode solve --image-path test_captcha.png

# Solve a single image with specific preprocessing method and solution method
python main.py --mode solve --image-path test_captcha.png --preprocessing-method precise_mathematical --solution-method crnn --use-gpu force

# Solve with raw preprocessing
python main.py --mode solve --image-path test_captcha.png --preprocessing-method raw

# Solve all PNG images in a directory
python main.py --mode solve --image-path captcha_dataset --use-gpu auto

# Solve with specific confidence threshold
python main.py --mode solve --image-path test_captcha.png --confidence-threshold 0.90

# Fast CAPTCHA solving with minimal preprocessing (optimized for speed)
python captcha_solver.py test_captcha.png

# Fast CAPTCHA solving multiple images in one execution (batch processing)
python captcha_solver.py test_captcha_3.png test_captcha_4.png

# Fast CAPTCHA solving all PNG files in current directory
python captcha_solver.py *.png
```

### Fast CAPTCHA Solver (captcha_solver.py)
The `captcha_solver.py` script is an optimized, independent solver designed for maximum speed, using CNRR Raw model:
- **Minimal preprocessing**: Only converts to grayscale and normalizes (no image resizing or complex operations)
- **GPU acceleration**: Automatically uses PyTorch CUDA when available for faster model loading
- **Model caching**: Caches the model in memory during execution for faster subsequent solves
- **Batch processing**: Can solve multiple images in a single execution for improved performance
- **Clean output**: Prints only the solution to stdout, with detailed timing in log file
- **Performance**: ~0.7 seconds total solve time (0.2s model load + 0.5s prediction) with GPU for first image, much faster for subsequent images in batch

### Model Performance Notes

- **CRNN Model**: Very fast and accurate models, except grayscale model. Other three models solve captchas with above 99.9% accuracy, in ~0.7 seconds, when used once, and under ~0.07 seconds when used in batch, using cached preloaded model. It's the recommended method for production use.
- **Segmentation Model**: Less reliable for ECI CAPTCHAs with accuracy (85%+). 
- **Fallback Sequence**: The default mode uses a sequence of methods (CRNN → segmentation → OCR) which provides the best overall reliability.
- **Fast Solver**: The `captcha_solver.py` script provides the fastest solving time for simple use cases where maximum accuracy is not critical.

For best results, use the segmentation method or the default mode which will automatically fall back to segmentation if other methods fail.

### 7. Integration with Other Systems
```python
# Integration with Phase 1
from eci_captcha_solver.captcha_solver import solve_captcha

# For a single image:
solution = solve_captcha("test_captcha_2.png")
print(f"Single image solution: {solution}")

# For multiple images (batch processing):
image_list = ["test_captcha_3.png", "test_captcha_2.png"]
solutions = [solve_captcha(img_path) for img_path in image_list]
for img, sol in zip(image_list, solutions):
	print(f"{img}: {sol}")
```

## 🔧 Configuration

Edit `config/captcha_config.json` for:
- Image dimensions (200x80)
- Confidence thresholds
- GPU usage settings
- Training parameters

## 🌐 API Service

This CAPTCHA solver is also available as a web API service for easy integration:

- **API Implementation**: `captcha_api.py` (FastAPI)
- **Documentation**: See `API_README.md` for detailed information
- **Endpoints**: `/solve`, `/health`, `/stats`
- **Deployment**: Multiple **completely free** options available (see `API_DEPLOYMENT.md`)
- **Free Deployment Summary**: See `FREE_DEPLOYMENT_SUMMARY.md` for a complete overview of all free options

### API Files
- `captcha_api.py` - Main API implementation
- `api_requirements.txt` - API dependencies
- `API_README.md` - API documentation
- `API_PLAN.md` - Implementation plan
- `test_api.py` - Testing script
- `.github/workflows/deploy-api.yml` - GitHub Actions deployment
- `render.yaml` - Render.com deployment configuration
- `API_DEPLOYMENT.md` - Deployment instructions
- `CLIENT_IMPLEMENTATION.md` - Client-side usage examples
- `FREE_DEPLOYMENT_SUMMARY.md` - Complete summary of free deployment options

### Deployment Options
1. **GitHub Actions with Self-Hosted Runner** - Optimal performance, free
2. **Render.com** - Cloud deployment, free tier available
3. **Railway.app** - Alternative cloud platform, free tier
4. **Fly.io** - Another cloud option, free tier

### Client-Side Integration
See `CLIENT_IMPLEMENTATION.md` for detailed examples in:
- JavaScript (with fallback strategy)
- Python (with fallback strategy)

## 📊 Expected Performance
- **Accuracy**: 99% on ECI CAPTCHAs
- **Speed**: 50-100ms per CAPTCHA (fast solver), 1-2 seconds (full solver)
- **Memory**: ~500MB loaded models
- **Hardware**: RTX 500 Ada optimized

## 🔗 Phase 1 Integration

Replace AntiCaptcha calls:
```python
# OLD
solution = anticaptcha_api.solve(image_path)

# NEW (Recommended: use command line interface)
# Solve a single image:
python captcha_solver.py test_captcha_2.png

# Solve multiple images (batch processing):
python captcha_solver.py test_captcha_1.png test_captcha_2.png

# If you need to call from Python, you can import solve_captcha from captcha_solver.py:

from captcha_solver import solve_captcha

# For a single image:
solution = solve_captcha("test_captcha_2.png")
print(f"Single image solution: {solution}")

# For multiple images (batch processing):
image_list = ["test_captcha_3.png", "test_captcha_2.png"]
solutions = [solve_captcha(img_path) for img_path in image_list]
for img, sol in zip(image_list, solutions):
	print(f"{img}: {sol}")
# Note: This only works if captcha_solver.py is in your PYTHONPATH and all dependencies are installed.
```

## 🧪 Testing
```bash
python test_framework.py --test-type all --save-results
```

## 📈 Key Features
- Optimized for 200x80 ECI images
- Text-preserving noise removal
- Ensemble model approach
- GPU acceleration
- Continuous learning
- Production monitoring
- Multiple preprocessing methods (grayscale, light_smoothing, precise_mathematical, raw)
- Unified CRNN model architecture (SimpleCRNN)
- Segmentation models
- Fast independent solver (`captcha_solver.py`)
- Batch processing capabilities

## 🛠️ Hardware Requirements
- **Minimum**: 8GB RAM, any GPU
- **Recommended**: 16GB+ RAM, RTX 500+ GPU
- **Optimal**: Your setup (32GB RAM, RTX 500 Ada)

## 📝 License
Internal use for ECI data collection project.