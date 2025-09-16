# Fast CAPTCHA Solver - captcha_solver.py

## Overview

The `captcha_solver.py` script is an optimized, independent solver designed for maximum speed and minimal overhead. Unlike the full `main.py` which provides comprehensive preprocessing and multiple solving methods, this script focuses on fast execution with minimal processing.

## Key Features

- **Minimal Preprocessing**: Only converts images to grayscale and normalizes pixel values (no resizing or complex operations)
- **GPU Acceleration**: Automatically uses PyTorch CUDA when available for faster model loading
- **Model Caching**: Caches the model in memory during execution for faster subsequent solves (within single execution)
- **Clean Output**: Prints only the solution to stdout, with detailed timing information logged to file
- **Performance**: ~0.7 seconds total solve time (0.2s model load + 0.5s prediction) with GPU
- **Batch Processing**: Can solve multiple images in a single execution

## Usage

```bash
python captcha_solver.py <image_path1> <image_path2> <image_path3> ...
```

Examples:
```bash
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

## Requirements

- The input image must be in the correct format (200x80 pixels, green text on white background)
- The image should not require any preprocessing (no lines, noise, etc.)
- PyTorch with CUDA support recommended for best performance

## Performance Characteristics

### Timing Breakdown
- **Model Loading**: ~0.2 seconds (with GPU acceleration)
- **Image Loading**: ~0.0 seconds
- **Preprocessing**: ~0.0 seconds (minimal operations)
- **Prediction**: ~0.5 seconds (neural network inference)
- **Total**: ~0.7 seconds

### GPU vs CPU Performance
- **GPU (PyTorch CUDA)**: Model loading ~0.2s
- **CPU (TensorFlow)**: Model loading ~0.5s

### Batch Processing Benefits
- **Model Caching**: Model is loaded only once for multiple images
- **Speed Improvement**: Second and subsequent images in a batch are much faster
- **Resource Efficiency**: Reduced GPU/CPU initialization overhead

## Implementation Details

### Minimal Preprocessing
The script performs only essential image processing:
1. Load the image using OpenCV
2. Convert to grayscale if needed
3. Normalize pixel values to [0, 1] range
4. Add required dimensions for model input

No resizing, no line removal, no noise reduction - the image is assumed to be in the correct format already.

### GPU Acceleration
When PyTorch with CUDA is available:
- Uses PyTorch for GPU-accelerated model loading
- Reduces model loading time from ~0.5s to ~0.2s
- Automatically falls back to CPU if GPU is not available

### Model Caching
Within a single execution:
- Model is loaded only once
- Subsequent solves reuse the cached model
- Cache expires after 10 minutes of inactivity

### Clean Output
- Terminal output contains only the solution (e.g., "39ks2h")
- One solution per line for multiple images
- All timing and diagnostic information goes to `captcha_solver.log`
- No TensorFlow warnings or other diagnostic messages in terminal

## Comparison with main.py

| Feature | captcha_solver.py | main.py |
|---------|-------------------|----------------|
| Preprocessing | Minimal (grayscale + normalize) | Comprehensive (multiple methods) |
| Speed | ~0.7 seconds | ~1-2 seconds |
| Accuracy | Dependent on input quality | High (multiple fallbacks) |
| GPU Support | PyTorch CUDA | TensorFlow GPU |
| Output | Clean (solution only) | Verbose (detailed info) |
| Batch Processing | Yes | Limited |
| Use Case | Fast batch processing | Production reliability |

## When to Use

Use `captcha_solver.py` when:
- You need maximum speed for batch processing
- Your images are already in the correct format
- You don't need advanced preprocessing
- You want clean output without diagnostic messages

Use `main.py` when:
- You need maximum accuracy
- Your images may require preprocessing
- You want detailed diagnostic information
- You're solving individual CAPTCHAs in production

## Troubleshooting

### No Solution Found
If the script outputs "no_solution_found":
- Check that the input image is in the correct format (200x80)
- Verify the image doesn't require preprocessing
- Ensure the model file exists at `models/simple_crnn_200x80_raw.keras`

### Slow Performance
If performance is slower than expected:
- Verify PyTorch with CUDA is installed (`python -c "import torch; print(torch.cuda.is_available())"`)
- Check that GPU memory is available
- Ensure no other processes are using the GPU

### Logging
Detailed timing and diagnostic information is logged to `captcha_solver.log`. Check this file for:
- Model loading times
- GPU vs CPU usage
- Detailed timing breakdown for each image
- Error messages and stack traces

## Batch Processing Examples

### Processing Multiple Images
```bash
# Process several images in one command
python captcha_solver.py image1.png image2.png image3.png

# Output:
# 39ks2h
# yk46kh
# abc123
```

### Processing Images with Spaces in Paths
```bash
# Use quotes for paths with spaces
python captcha_solver.py "image with spaces.png" "another image.png"
```

### Processing All PNG Files in Directory
```bash
# Process all PNG files in current directory
python captcha_solver.py *.png
```

### Performance Benefits of Batch Processing
When processing multiple images in a single execution:
1. Model is loaded only once (first image takes ~0.7s)
2. Subsequent images reuse the cached model (~0.05s for model loading)
3. Significant time savings for large batches