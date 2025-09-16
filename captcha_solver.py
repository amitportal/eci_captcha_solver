#!/usr/bin/env python3
"""
captcha_solver.py - Independent ECI CAPTCHA Solver
Loads the raw model and generates solutions quickly without preprocessing
"""

# Suppress TensorFlow messages before importing
import os
import sys
import contextlib
import glob

# Redirect stderr to suppress TensorFlow messages
stderr_original = sys.stderr
sys.stderr = open(os.devnull, 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN optimizations

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import time
from datetime import datetime
from pathlib import Path
import argparse

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch CUDA available with {torch.cuda.device_count()} device(s)", file=sys.stderr)
except ImportError:
    TORCH_AVAILABLE = False

# Restore stderr after TensorFlow imports
sys.stderr = stderr_original

# Set TensorFlow logging levels
tf.get_logger().setLevel('ERROR')

# Additional TensorFlow logging suppression
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Enable unsafe deserialization for Keras models
keras.config.enable_unsafe_deserialization()

# Setup logging to file only (no console output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("captcha_solver.log")
    ]
)
logger = logging.getLogger(__name__)

# Character set definitions
LOWERCASE = ''.join([c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in {'i', 'l', 'o'}])
DIGITS = ''.join([c for c in '0123456789' if c not in {'1', '0'}])
CHARACTERS = LOWERCASE + DIGITS
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

# Global variables for model caching (within single execution)
_model = None
_last_used = None
CACHE_DURATION = 600  # 10 minutes in seconds

def get_device_info():
    """Get information about the device being used (GPU/CPU)"""
    if TORCH_AVAILABLE:
        return f"PyTorch-GPU ({torch.cuda.device_count()} devices)"
    elif tf.config.list_physical_devices('GPU'):
        gpu_devices = tf.config.list_physical_devices('GPU')
        return f"TF-GPU ({len(gpu_devices)} devices)"
    else:
        return "CPU"

def load_model():
    """Load the model with caching to avoid reloading within single execution"""
    global _model, _last_used
    
    current_time = time.time()
    
    # Check if we have a cached model that's still valid
    if _model is not None and _last_used is not None:
        if current_time - _last_used < CACHE_DURATION:
            logger.info(f"Using cached model (age: {current_time - _last_used:.2f}s)")
            _last_used = current_time  # Update last used time
            return _model
        else:
            logger.info(f"Cache expired (age: {current_time - _last_used:.2f}s > {CACHE_DURATION}s)")
    
    # Load the model
    load_start_time = time.time()
    logger.info("Loading model...")
    try:
        # Always load the model from the 'models/simple_crnn_200x80_raw.keras' path relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "simple_crnn_200x80_raw.keras")
        model_load_start = time.time()
        _model = keras.models.load_model(model_path, compile=False)
        model_load_time = time.time() - model_load_start
        
        _last_used = current_time
        load_time = time.time() - load_start_time
        device_info = get_device_info()
        logger.info(f"Model loaded successfully in {load_time:.3f}s (model load: {model_load_time:.3f}s, using {device_info})")
        return _model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def solve_captcha(image_path):
    """Solve a CAPTCHA image and return the solution"""
    overall_start_time = time.time()
    
    # Load the model
    model_load_start = time.time()
    model = load_model()
    model_load_time = time.time() - model_load_start
    
    if model is None:
        logger.error("Model could not be loaded")
        print("no_solution_found")
        return
    
    try:
        # Load image directly without any preprocessing
        image_load_start = time.time()
        image = cv2.imread(image_path)
        image_load_time = time.time() - image_load_start
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to the exact format expected by the model (200x80 grayscale normalized)
        # No resizing, no processing - assume image is already correct format
        process_start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        process_time = time.time() - process_start
        
        # Make prediction
        prediction_start = time.time()
        prediction = model.predict(image, verbose=0)
        prediction_time = time.time() - prediction_start
        
        # Process prediction to get characters
        processing_start = time.time()
        predicted_chars = []
        confidence_scores = []
        
        # For each character position
        for i in range(6):  # CAPTCHA_LENGTH = 6
            # Get the character probabilities for this position
            char_probs = prediction[0, i, :]
            # Get the most likely character
            char_idx = np.argmax(char_probs)
            char_confidence = char_probs[char_idx]
            
            # Convert index to character
            if char_idx < len(NUM_TO_CHAR):
                predicted_chars.append(NUM_TO_CHAR[int(char_idx)])
                confidence_scores.append(char_confidence)
            else:
                predicted_chars.append('?')
                confidence_scores.append(0.0)
        
        # Join characters to form solution
        solution = ''.join(predicted_chars)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        processing_time = time.time() - processing_start
        
        # Check if solution is valid (all characters recognized)
        is_valid = all(c != '?' for c in predicted_chars)
        
        # Calculate time taken
        total_time = time.time() - overall_start_time
        
        # Log detailed timing information
        status = "solved" if is_valid else "no_solution"
        device_info = get_device_info()
        logger.info(f"Solution: {solution}, "
                   f"Total: {total_time:.3f}s, "
                   f"Model load: {model_load_time:.3f}s, "
                   f"Image load: {image_load_time:.3f}s, "
                   f"Processing: {process_time:.3f}s, "
                   f"Prediction: {prediction_time:.3f}s, "
                   f"Result processing: {processing_time:.3f}s, "
                   f"Confidence: {avg_confidence:.3f}, "
                   f"Device: {device_info}, "
                   f"Status: {status}")
        
        # Print result to terminal
        if is_valid:
            return solution
        else:
            return "no_solution_found"
            
    except Exception as e:
        total_time = time.time() - overall_start_time
        logger.error(f"no_solution,{total_time:.3f}s,0.0,error: {str(e)}")
        return "no_solution_found"

def expand_paths(image_paths):
    """Expand glob patterns and handle directories"""
    expanded_paths = []
    for path in image_paths:
        # Handle glob patterns
        if '*' in path or '?' in path:
            matched_paths = glob.glob(path)
            if matched_paths:
                expanded_paths.extend(matched_paths)
            else:
                # If glob doesn't match anything, add the original path
                expanded_paths.append(path)
        # Handle directories
        elif os.path.isdir(path):
            # Add all PNG files in the directory
            dir_pngs = glob.glob(os.path.join(path, "*.png"))
            if dir_pngs:
                expanded_paths.extend(dir_pngs)
            else:
                # If no PNG files found, add the directory path (will show error)
                expanded_paths.append(path)
        else:
            # Regular file path
            expanded_paths.append(path)
    return expanded_paths

def main():
    """Main function to parse arguments and solve CAPTCHA"""
    parser = argparse.ArgumentParser(
        description='Solve ECI CAPTCHA images quickly and cheaply.',
        epilog="""
Required:
    Path(s) to the ECI CAPTCHA image(s) to solve.

Usage examples:
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

Arguments:
    image_paths (str): One or more paths to ECI CAPTCHA image files or directories to be solved.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image_paths', nargs='+', help='Path(s) to the ECI CAPTCHA image(s) or directories to solve')

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    try:
        args = parser.parse_args()
    except SystemExit:
        # If there's an error in parsing, show help and exit
        parser.print_help()
        sys.exit(1)
    
    # Expand paths to handle glob patterns and directories
    expanded_paths = expand_paths(args.image_paths)
    
    # Process each image
    for image_path in expanded_paths:
        solution = solve_captcha(image_path)
        print(solution)

if __name__ == "__main__":
    main()