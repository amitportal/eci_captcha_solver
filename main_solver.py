#!/usr/bin/env python3
"""
main_solver.py - ECI CAPTCHA Solver Core Implementation
Updated for 200x80 images with improved text preservation preprocessing
"""

import os
import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
import argparse
import logging
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import string
import random
from datetime import datetime
import base64
from PIL import Image
import io
import time
import gc
import sys
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Remove all handlers associated with the logger object (avoid duplicate logs)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Remove all handlers associated with the logger object (avoid duplicate logs)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Import GPU acceleration
try:
    from gpu_accelerator import get_accelerator, print_gpu_status
    GPU_ACCELERATOR_AVAILABLE = True
    logger.info("✓ GPU accelerator module imported successfully")
except ImportError as e:
    logger.warning(f"GPU accelerator not available - using CPU only: {e}")
    GPU_ACCELERATOR_AVAILABLE = False
    get_accelerator = None
    print_gpu_status = None

class OptimizedCaptchaConfig:
    """Optimized configuration for ECI CAPTCHAs - Updated for 200x80 images"""
    
    # Character set (prioritized based on frequency analysis)
    LOWERCASE = ''.join([c for c in string.ascii_lowercase if c not in {'i', 'l', 'o'}])  # 72.7% of dataset
    DIGITS = ''.join([c for c in string.digits if c not in {'1', '0'}])              # 25.9% of dataset
    
    # Optimized character set (frequency-based ordering)
    CHARACTERS = LOWERCASE + DIGITS
    CHAR_TO_NUM = {char: idx for idx, char in enumerate(CHARACTERS)}
    NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
    
    # CAPTCHA properties (confirmed from analysis)
    CAPTCHA_LENGTH = 6
    NUM_CLASSES = len(CHARACTERS)  # 31 classes
    
    # Updated image properties for ECI CAPTCHAs (200x80 actual size)
    IMG_WIDTH = 200   # Actual ECI CAPTCHA width
    IMG_HEIGHT = 80   # Actual ECI CAPTCHA height
    
    # Training parameters (optimized for mixed-case patterns)
    BATCH_SIZE = 32
    EPOCHS = 50       # Reduced due to pattern consistency
    LEARNING_RATE = 0.0005  # Lower for better convergence
    
    # Ensemble parameters
    CONFIDENCE_THRESHOLD = 0.80  # Lower due to good data quality
    
    # GPU configuration
    USE_GPU_FOR_TRAINING = True   # Force GPU for training
    USE_GPU_FOR_INFERENCE = "auto"  # Auto-detect for inference

class NanDetectionCallback(keras.callbacks.Callback):
    """Custom callback to detect NaN values during training and take corrective actions"""
    
    def __init__(self):
        super().__init__()
        self.nan_detected = False
        
    def on_batch_end(self, batch, logs=None):
        """Check for NaN values after each batch"""
        if logs:
            loss = logs.get('loss')
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    logger.error(f"NaN/Inf detected in loss at batch {batch}: loss={loss}")
                    self.nan_detected = True
                    # Stop training
                    self.model.stop_training = True
                    
    def on_epoch_end(self, epoch, logs=None):
        """Log epoch metrics and check for issues"""
        if logs:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            accuracy = logs.get('accuracy', 0)
            val_accuracy = logs.get('val_accuracy', 0)
            
            logger.info(f"Epoch {epoch+1}: loss={loss:.4f}, accuracy={accuracy:.4f}, "
                       f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
            
            # Check for exploding gradients or other issues
            if loss > 1000:
                logger.warning(f"Very high loss detected: {loss}")
                
            if val_loss > 1000:
                logger.warning(f"Very high validation loss detected: {val_loss}")

class TrainingMetricsLogger(keras.callbacks.Callback):
    """Custom callback to log detailed training metrics and save them to files"""
    
    def __init__(self):
        super().__init__()
        self.metrics_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
    def on_epoch_end(self, epoch, logs=None):
        """Log detailed metrics at the end of each epoch"""
        if logs:
            # Store metrics
            self.metrics_history['epoch'].append(epoch + 1)
            self.metrics_history['loss'].append(logs.get('loss', 0))
            self.metrics_history['accuracy'].append(logs.get('accuracy', 0))
            self.metrics_history['val_loss'].append(logs.get('val_loss', 0))
            self.metrics_history['val_accuracy'].append(logs.get('val_accuracy', 0))
            
            # Get current learning rate
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                lr_value = lr.numpy()
            else:
                lr_value = lr
            self.metrics_history['learning_rate'].append(lr_value)
            
            # Detailed logging
            logger.info(f"Epoch {epoch+1:2d}: "
                       f"loss={logs.get('loss', 0):.4f}, "
                       f"acc={logs.get('accuracy', 0):.4f}, "
                       f"val_loss={logs.get('val_loss', 0):.4f}, "
                       f"val_acc={logs.get('val_accuracy', 0):.4f}, "
                       f"lr={lr_value:.2e}")
            
            # Log improvement indicators
            if epoch > 0:
                prev_val_acc = self.metrics_history['val_accuracy'][-2]
                curr_val_acc = self.metrics_history['val_accuracy'][-1]
                if curr_val_acc > prev_val_acc:
                    improvement = curr_val_acc - prev_val_acc
                    logger.info(f"          ↳ Validation accuracy improved by {improvement:.4f}")
                    
                prev_val_loss = self.metrics_history['val_loss'][-2]
                curr_val_loss = self.metrics_history['val_loss'][-1]
                if curr_val_loss < prev_val_loss:
                    improvement = prev_val_loss - curr_val_loss
                    logger.info(f"          ↳ Validation loss improved by {improvement:.4f}")

# GPU acceleration helper functions (using new GPU accelerator system)
def setup_tensorflow_gpu():
    """Setup TensorFlow GPU if available (for compatibility)"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Configure memory growth to avoid allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"TensorFlow GPU configured: {len(gpus)} GPU(s)")
            return True
        return False
    except Exception as e:
        logger.debug(f"TensorFlow GPU setup skipped: {e}")
        return False

def get_gpu_memory_info():
    """Get GPU memory information using GPU accelerator"""
    try:
        if GPU_ACCELERATOR_AVAILABLE and get_accelerator:
            accelerator = get_accelerator()
            info = accelerator.get_device_info()
            
            if info["backend"] == "pytorch" and info.get("gpu_count", 0) > 0:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.get_device_properties(0)
                    total_memory = device.total_memory // (1024**2)  # MB
                    allocated = torch.cuda.memory_allocated(0) // (1024**2)  # MB
                    cached = torch.cuda.memory_reserved(0) // (1024**2)  # MB
                    return {
                        'total': total_memory,
                        'allocated': allocated,
                        'cached': cached,
                        'free': total_memory - cached
                    }
        return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}
    except Exception:
        return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}

class AdvancedPreprocessor:
    """Advanced preprocessing optimized for ECI CAPTCHA patterns with text preservation"""
    
    def __init__(self, use_gpu: bool = True, preprocessing_method: str = "light_smoothing"):
        self.target_size = (OptimizedCaptchaConfig.IMG_WIDTH, OptimizedCaptchaConfig.IMG_HEIGHT)
        self.preprocessing_method = preprocessing_method
        
        # Optimized kernels for ECI CAPTCHAs
        self.noise_removal_kernel = np.ones((2, 2), np.uint8)
        self.text_preservation_kernel = np.ones((3, 3), np.uint8)
        self.sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        # Initialize GPU acceleration if available
        self.use_gpu = use_gpu and GPU_ACCELERATOR_AVAILABLE
        if self.use_gpu and get_accelerator is not None:
            try:
                self.accelerator = get_accelerator()
                self.gpu_ops = self.accelerator.create_tensor_ops()
                logger.info("✓ GPU acceleration enabled for preprocessing")
            except Exception as e:
                logger.warning(f"GPU acceleration failed, using CPU: {e}")
                self.use_gpu = False
                self.accelerator = None
                self.gpu_ops = None
        else:
            self.accelerator = None
            self.gpu_ops = None
    
    def light_smoothing(self, image: np.ndarray) -> np.ndarray:
        """Apply light smoothing to the image - default method"""
        try:
            # Apply bilateral filter for noise reduction while preserving edges
            smoothed = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Apply mild median blur to reduce salt-and-pepper noise
            smoothed = cv2.medianBlur(smoothed, 3)
            
            return smoothed
        except Exception as e:
            logger.warning(f"Light smoothing failed: {e}")
            return image
    
    def precise_mathematical_preprocessing(self, image_path: str) -> np.ndarray:
        """Apply the precise mathematical preprocessing approach from preprocessing_pixel.py"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            h, w = img.shape[:2]
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Color thresholds from ultra-precise analysis
            GREEN_HSV_LOWER = [35, 20, 20]
            GREEN_HSV_UPPER = [90, 255, 255]
            
            HORIZONTAL_LINES = [
                {"name": "Yellow", "y_center": 30, "hsv_lower": [10, 30, 100], "hsv_upper": [50, 255, 255]},
                {"name": "Cyan", "y_center": 40, "hsv_lower": [80, 30, 100], "hsv_upper": [100, 255, 255]},
                {"name": "Pink", "y_center": 50, "hsv_lower": [140, 30, 100], "hsv_upper": [170, 255, 255]}
            ]
            
            # Diagonal line parameters
            DIAGONAL_SLOPE = 0.393910
            DIAGONAL_BUFFER = 2
            SEARCH_DISTANCE = 20
            
            def hsv_mask(hsv, lower, upper):
                """Create HSV mask with given range"""
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                return cv2.inRange(hsv, lower, upper)
            
            def create_horizontal_line_mask(h, w, y_center, buffer=2):
                """Create a mask for horizontal line with buffer"""
                mask = np.zeros((h, w), dtype=np.uint8)
                y_start = max(0, y_center - buffer)
                y_end = min(h, y_center + buffer + 1)
                mask[y_start:y_end, :] = 255
                return mask
            
            def create_precise_diagonal_mask(h, w, slope, buffer=2):
                """Create a mask for the diagonal line using ultra-precise equation"""
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Ultra-precise diagonal equation: y = (h-1) - slope * x
                for x in range(w):
                    y_center = int((h - 1) - slope * x)
                    # Add buffer around the line
                    for dy in range(-buffer, buffer + 1):
                        y = y_center + dy
                        if 0 <= y < h:
                            mask[y, x] = 255
                            
                return mask
            
            # Create green text mask
            green_mask = hsv_mask(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
            
            # Process horizontal lines
            all_horizontal_pixels = np.zeros((h, w), dtype=np.uint8)
            
            for line_info in HORIZONTAL_LINES:
                name = line_info['name']
                y_center = line_info['y_center']
                lower = line_info['hsv_lower']
                upper = line_info['hsv_upper']
                
                # Create color mask
                color_mask = hsv_mask(hsv, lower, upper)
                
                # Create region mask
                region_mask = create_horizontal_line_mask(h, w, y_center)
                
                # Find pixels in both color and region
                line_pixels = cv2.bitwise_and(color_mask, region_mask)
                
                # Add to combined horizontal mask
                all_horizontal_pixels = cv2.bitwise_or(all_horizontal_pixels, line_pixels)
            
            # Create diagonal line mask
            diagonal_mask = create_precise_diagonal_mask(h, w, DIAGONAL_SLOPE, DIAGONAL_BUFFER)
            
            # Find diagonal green pixels
            diagonal_green_pixels = cv2.bitwise_and(green_mask, diagonal_mask)
            
            # Combine all line pixels
            all_line_pixels = cv2.bitwise_or(all_horizontal_pixels, diagonal_green_pixels)
            
            # Process all line pixels
            result = img.copy()
            
            # Get coordinates of all line pixels
            ys, xs = np.where(all_line_pixels > 0)
            
            for y, x in zip(ys, xs):
                # Look for nearby green text pixels
                found_text = False
                fill_val = None
                
                # Check above (up to SEARCH_DISTANCE pixels)
                for y_offset in range(1, min(SEARCH_DISTANCE + 1, y + 1)):
                    y_check = y - y_offset
                    if y_check >= 0 and green_mask[y_check, x] > 0:
                        fill_val = img[y_check, x]
                        found_text = True
                        break
                        
                # If not found above, check below
                if not found_text:
                    for y_offset in range(1, min(SEARCH_DISTANCE + 1, h - y)):
                        y_check = y + y_offset
                        if y_check < h and green_mask[y_check, x] > 0:
                            fill_val = img[y_check, x]
                            found_text = True
                            break
                
                # Fill or erase
                if found_text and fill_val is not None:
                    result[y, x] = fill_val
                else:
                    result[y, x] = [255, 255, 255]  # White
            
            # Apply post-processing smoothing
            result = cv2.bilateralFilter(result, 5, 75, 75)
            
            return result
            
        except Exception as e:
            logger.warning(f"Precise mathematical preprocessing failed: {e}")
            return cv2.imread(str(image_path))  # Return original image as fallback
    
    def preserve_green_text_during_line_removal(self, image: np.ndarray) -> np.ndarray:
        """Remove colored lines while preserving green text integrity"""
        try:
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define precise color ranges for ECI CAPTCHA elements
            # Green text range (preserve - this is our target text)
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # White/light background range (preserve)
            white_lower = np.array([0, 0, 180])
            white_upper = np.array([180, 25, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # Yellow line range (remove)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            # Cyan line range (remove)
            cyan_lower = np.array([85, 100, 100])
            cyan_upper = np.array([95, 255, 255])
            cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)
            
            # Pink/Magenta line range (remove)
            pink_lower = np.array([140, 100, 100])
            pink_upper = np.array([170, 255, 255])
            pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
            
            # Combine noise masks (what to remove)
            noise_mask = cv2.bitwise_or(yellow_mask, cyan_mask)
            noise_mask = cv2.bitwise_or(noise_mask, pink_mask)
            
            # CRITICAL: Remove noise ONLY where it doesn't intersect with green text
            # This prevents fragmenting the text characters
            text_protected_noise = cv2.bitwise_and(noise_mask, cv2.bitwise_not(green_mask))
            
            # Combine preservation masks (what to keep)
            preserve_mask = cv2.bitwise_or(green_mask, white_mask)
            
            # Apply morphological operations to clean up while preserving text
            preserve_mask = cv2.morphologyEx(preserve_mask, cv2.MORPH_CLOSE, self.text_preservation_kernel)
            
            # Create clean image
            result = np.full_like(image, (255, 255, 255))  # White background
            result[preserve_mask > 0] = image[preserve_mask > 0]
            
            # Fill removed noise areas with white (but not where text exists)
            result[text_protected_noise > 0] = [255, 255, 255]
            
            return result
            
        except Exception as e:
            logger.warning(f"Text-preserving noise removal failed: {e}")
            return image
    
    def remove_diagonal_lines_carefully(self, image: np.ndarray) -> np.ndarray:
        """Remove diagonal lines while preserving text characters"""
        try:
            # Convert to grayscale for line detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create a mask for text areas (darker regions)
            _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Dilate text mask to create protection zones around characters
            text_protection = cv2.dilate(text_mask, self.text_preservation_kernel, iterations=2)
            
            # Detect edges for line detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Remove edges that intersect with text protection zones
            protected_edges = cv2.bitwise_and(edges, cv2.bitwise_not(text_protection))
            
            # Detect lines only in non-text areas
            lines = cv2.HoughLines(protected_edges, 1, np.pi/180, threshold=60)
            
            if lines is not None:
                for rho, theta in lines[0]:
                    # Filter for diagonal lines (not horizontal/vertical)
                    angle = theta * 180 / np.pi
                    if 20 < angle < 70 or 110 < angle < 160:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        
                        # Create line mask
                        line_mask = np.zeros_like(gray)
                        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
                        
                        # Only remove line where it doesn't intersect with text
                        safe_removal_mask = cv2.bitwise_and(line_mask, cv2.bitwise_not(text_protection))
                        
                        # Apply removal
                        image[safe_removal_mask > 0] = [255, 255, 255]
            
            return image
            
        except Exception as e:
            logger.warning(f"Careful diagonal line removal failed: {e}")
            return image
    
    def enhance_text_clarity(self, image: np.ndarray) -> np.ndarray:
        """Enhanced text clarity for ECI CAPTCHAs"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Adaptive histogram equalization with reduced clip limit to preserve text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Bilateral filter for noise reduction while preserving edges
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Gentle sharpening to enhance character edges
            sharpened = cv2.filter2D(filtered, -1, self.sharpening_kernel)
            
            # Ensure good contrast but don't over-process
            normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Text enhancement failed: {e}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def preprocess_image(self, image_path: str, save_preprocessed: bool = False, output_dir: Optional[str] = None) -> np.ndarray:
        """Complete optimized preprocessing pipeline for 200x80 images with enhanced validation"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                # Return blank image as fallback
                return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
            
            # Verify dimensions and resize if necessary
            if image.shape[:2] != (self.target_size[1], self.target_size[0]):
                # Use GPU acceleration for resizing if available
                if self.use_gpu and self.gpu_ops:
                    try:
                        # Convert to grayscale first for GPU processing
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        resized = self.gpu_ops.resize_batch(gray, self.target_size)
                        image = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    except Exception as e:
                        logger.debug(f"GPU resize failed, using CPU: {e}")
                        image = cv2.resize(image, self.target_size)
                else:
                    image = cv2.resize(image, self.target_size)
                logger.debug(f"Resized image from {image_path}")
            
            # Apply the selected preprocessing method
            if self.preprocessing_method == "precise_mathematical":
                # Use the precise mathematical approach
                image = self.precise_mathematical_preprocessing(image_path)
            elif self.preprocessing_method == "light_smoothing":
                # Apply light smoothing (default)
                image = self.light_smoothing(image)
            elif self.preprocessing_method == "grayscale":
                # Default preprocessing pipeline (original approach)
                # Step 1: Remove colored lines while preserving green text
                image = self.preserve_green_text_during_line_removal(image)
                
                # Step 2: Carefully remove diagonal lines
                image = self.remove_diagonal_lines_carefully(image)
                
                # Step 3: Enhance text clarity
                processed = self.enhance_text_clarity(image)
                
                # Save preprocessed image for analysis if requested
                if save_preprocessed and output_dir:
                    try:
                        # Create output directory if it doesn't exist
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        # Get original filename and create output filename
                        original_name = Path(image_path).stem
                        output_filename = f"{original_name}_preprocessed.png"
                        output_file = output_path / output_filename
                        
                        # Save the preprocessed image (before normalization)
                        cv2.imwrite(str(output_file), processed)
                        logger.debug(f"Saved preprocessed image: {output_file}")
                        
                    except Exception as e:
                        logger.debug(f"Failed to save preprocessed image: {e}")
                
                # Step 4: Normalize to [0, 1] - use GPU if available
                if self.use_gpu and self.gpu_ops:
                    try:
                        processed = self.gpu_ops.normalize_batch(processed.astype(np.float32), mean=0.0, std=255.0)
                    except Exception as e:
                        logger.debug(f"GPU normalization failed, using CPU: {e}")
                        processed = processed.astype(np.float32) / 255.0
                else:
                    processed = processed.astype(np.float32) / 255.0
                
                # Step 5: Add channel dimension for model input
                processed = np.expand_dims(processed, axis=-1)
                
                # Validate output
                if np.isnan(processed).any() or np.isinf(processed).any():
                    logger.error(f"NaN/Inf values detected in preprocessed image: {image_path}")
                    # Return blank image as fallback
                    return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
                
                return processed
            
            # If we used a specific preprocessing method, convert to the expected format
            if self.preprocessing_method in ["light_smoothing", "precise_mathematical", "grayscale"]:
                # Convert to grayscale
                if len(image.shape) == 3:
                    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    processed = image
                
                # Normalize to [0, 1]
                processed = processed.astype(np.float32) / 255.0
                
                # Add channel dimension
                processed = np.expand_dims(processed, axis=-1)
                
                # Validate output
                if np.isnan(processed).any() or np.isinf(processed).any():
                    logger.error(f"NaN/Inf values detected in preprocessed image: {image_path}")
                    # Return blank image as fallback
                    return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
                
                return processed
            
            # Default return (should not reach here)
            logger.warning(f"Unknown preprocessing method: {self.preprocessing_method}")
            return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_path}: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Return blank image as fallback
            return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)

class OptimizedDataLoader:
    """Optimized data loader for ECI CAPTCHA dataset with 200x80 images"""
    
    def __init__(self, dataset_dir: str, csv_file: str = None, use_gpu: bool = False, preprocessing_method: str = "grayscale"):
        self.dataset_dir = Path(dataset_dir)
        self.csv_file = csv_file
        self.use_gpu = use_gpu
        self.preprocessor = AdvancedPreprocessor(preprocessing_method=preprocessing_method)
    
    def load_from_csv(self) -> Tuple[List[str], List[str]]:
        """Load data from consolidated CSV with validation"""
        if not self.csv_file or not Path(self.csv_file).exists():
            logger.warning("CSV file not provided or not found, scanning directory")
            return self.load_from_directory()
        
        try:
            df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded CSV with {len(df)} entries")
            
            # Filter for valid 6-character solutions only
            valid_df = df[df['captcha_length'] == 6].copy()
            logger.info(f"Found {len(valid_df)} valid 6-character CAPTCHAs")
            
            image_paths = []
            solutions = []
            missing_files = 0
            invalid_images = 0
            invalid_solutions = 0
            
            for _, row in valid_df.iterrows():
                # Extract filename from source_json or use solution-based naming
                if 'source_json' in row and pd.notna(row['source_json']):
                    source_json = str(row['source_json'])
                    if '.json' in source_json:
                        filename = source_json.replace('.json', '.png')
                    else:
                        filename = f"{source_json}.png"
                else:
                    # Fallback filename generation
                    filename = f"{row['solution']}_captcha.png"
                
                image_path = self.dataset_dir / filename
                
                if image_path.exists():
                    # Validate image dimensions
                    img = cv2.imread(str(image_path))
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        # Validate solution format
                        solution = str(row['solution']).strip().lower()
                        if self._is_valid_solution(solution):
                            image_paths.append(str(image_path))
                            # Normalize solution to lowercase (all solutions are lowercase)
                            solutions.append(solution)
                        else:
                            invalid_solutions += 1
                            logger.debug(f"Invalid solution format: {solution}")
                    else:
                        invalid_images += 1
                else:
                    missing_files += 1
            
            logger.info(f"Found {len(image_paths)} valid image files")
            logger.info(f"Missing files: {missing_files}, Invalid images: {invalid_images}, Invalid solutions: {invalid_solutions}")
            
            return image_paths, solutions
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return self.load_from_directory()
    
    def load_from_directory(self) -> Tuple[List[str], List[str]]:
        """Fallback: load from directory scanning"""
        image_paths = []
        solutions = []
        invalid_solutions = 0
        
        for json_file in self.dataset_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                solution = data.get('solution', '').strip().lower()
                
                if len(solution) == OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                    png_file = json_file.with_suffix('.png')
                    if png_file.exists():
                        # Validate image
                        img = cv2.imread(str(png_file))
                        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                            # Validate solution format
                            if self._is_valid_solution(solution):
                                image_paths.append(str(png_file))
                                solutions.append(solution)
                            else:
                                invalid_solutions += 1
                                logger.debug(f"Invalid solution format in {json_file}: {solution}")
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        logger.info(f"Directory scan found {len(image_paths)} valid samples, invalid solutions: {invalid_solutions}")
        return image_paths, solutions
    
    def _is_valid_solution(self, solution: str) -> bool:
        """Validate that a solution contains only valid characters"""
        if len(solution) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
            return False
        
        # Check that all characters are in our character set
        valid_chars = set(OptimizedCaptchaConfig.CHARACTERS)
        return all(c in valid_chars for c in solution)
    
    def create_tf_dataset(self, image_paths: List[str], labels: List[str], 
                         batch_size: int = None, shuffle: bool = True,
                         cache: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset for 200x80 images with enhanced validation"""
        if batch_size is None:
            batch_size = OptimizedCaptchaConfig.BATCH_SIZE
        
        logger.info(f"Creating TF dataset with {len(image_paths)} samples, batch_size={batch_size}")
        
        def load_and_preprocess(image_path, label):
            # Load and preprocess image
            image = tf.py_function(
                func=lambda path: self._safe_preprocess_image(path.numpy().decode('utf-8')),
                inp=[image_path],
                Tout=tf.float32
            )
            # Set shape on the returned tensor, not the function
            image = tf.ensure_shape(image, [OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1])
            
            # Encode label with validation
            label_encoded = tf.py_function(
                func=lambda lbl: self._safe_encode_label(lbl.numpy().decode('utf-8')),
                inp=[label],
                Tout=tf.int32
            )
            label_encoded = tf.ensure_shape(label_encoded, [OptimizedCaptchaConfig.CAPTCHA_LENGTH])
            
            return image, label_encoded
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Add validation step to check for issues
        dataset = dataset.filter(self._validate_sample)
        
        if cache:
            dataset = dataset.cache()
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info("TF dataset created successfully")
        return dataset
    
    def _safe_preprocess_image(self, image_path: str) -> np.ndarray:
        """Safely preprocess image with error handling"""
        try:
            result = self.preprocessor.preprocess_image(image_path)
            # Check for NaN or infinite values
            if np.isnan(result).any() or np.isinf(result).any():
                logger.warning(f"NaN/Inf values detected in preprocessed image: {image_path}")
                # Return a blank image as fallback
                return np.zeros((OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1), dtype=np.float32)
            return result
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1), dtype=np.float32)
    
    def _safe_encode_label(self, label: str) -> np.ndarray:
        """Safely encode label with validation"""
        try:
            # Validate label length
            if len(label) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                raise ValueError(f"Invalid label length: expected {OptimizedCaptchaConfig.CAPTCHA_LENGTH}, got {len(label)}")
            
            # Validate characters
            for char in label:
                if char not in OptimizedCaptchaConfig.CHAR_TO_NUM:
                    raise ValueError(f"Invalid character in label: {char}")
            
            # Encode label
            encoded = np.array([OptimizedCaptchaConfig.CHAR_TO_NUM[c] for c in label])
            
            # Validate encoded values
            if np.any(encoded < 0) or np.any(encoded >= OptimizedCaptchaConfig.NUM_CLASSES):
                raise ValueError(f"Invalid encoded values: min={np.min(encoded)}, max={np.max(encoded)}")
                
            return encoded
        except Exception as e:
            logger.error(f"Error encoding label '{label}': {e}")
            # Return zeros as fallback
            return np.zeros(OptimizedCaptchaConfig.CAPTCHA_LENGTH, dtype=np.int32)
    
    def _validate_sample(self, image, label):
        """Validate a sample to filter out problematic data"""
        try:
            # Check for NaN or infinite values in image
            has_nan = tf.reduce_any(tf.math.is_nan(image))
            has_inf = tf.reduce_any(tf.math.is_inf(image))
            
            # Check label validity
            has_invalid_label = tf.reduce_any(tf.less(label, 0)) or tf.reduce_any(tf.greater_equal(label, OptimizedCaptchaConfig.NUM_CLASSES))
            wrong_label_length = tf.not_equal(tf.shape(label)[0], OptimizedCaptchaConfig.CAPTCHA_LENGTH)
            
            # Return True only if all validations pass
            return tf.logical_not(tf.logical_or(
                tf.logical_or(has_nan, has_inf),
                tf.logical_or(has_invalid_label, wrong_label_length)
            ))
        except Exception as e:
            logger.error(f"Error in sample validation: {e}")
            return False
    
    def prepare_datasets(self, test_size: float = 0.15, val_size: float = 0.15,
                        cache_datasets: bool = True) -> Dict:
        """Prepare optimized train/val/test datasets for 200x80 images with enhanced debugging"""
        logger.info("Preparing datasets...")
        image_paths, solutions = self.load_from_csv()
        
        # Allow smaller datasets for testing
        if len(image_paths) < 10:
            raise ValueError(f"Insufficient data: only {len(image_paths)} samples found")
        
        # Log dataset statistics
        logger.info(f"Total samples: {len(image_paths)}")
        logger.info(f"Unique solutions: {len(set(solutions))}")
        
        # Check for data quality issues
        logger.info("Checking data quality...")
        invalid_samples = 0
        for i, (path, solution) in enumerate(zip(image_paths, solutions)):
            if len(solution) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.warning(f"Invalid solution length at index {i}: {solution} (length: {len(solution)})")
                invalid_samples += 1
            elif not all(c in OptimizedCaptchaConfig.CHARACTERS for c in solution):
                logger.warning(f"Invalid characters in solution at index {i}: {solution}")
                invalid_samples += 1
        
        if invalid_samples > 0:
            logger.warning(f"Found {invalid_samples} invalid samples out of {len(image_paths)} total samples")
        
        # Stratified split to maintain character distribution
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, solutions, test_size=test_size, random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=val_size_adjusted, random_state=42
        )
        
        logger.info(f"Dataset split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        train_dataset = self.create_tf_dataset(train_paths, train_labels, shuffle=True, cache=cache_datasets)
        val_dataset = self.create_tf_dataset(val_paths, val_labels, shuffle=False, cache=cache_datasets)
        test_dataset = self.create_tf_dataset(test_paths, test_labels, shuffle=False, cache=False)
        
        logger.info("Datasets prepared successfully")
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels), 
            'test': (test_paths, test_labels),
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'stats': {
                'total_samples': len(image_paths),
                'train_samples': len(train_paths),
                'val_samples': len(val_paths),
                'test_samples': len(test_paths),
                'unique_solutions': len(set(solutions)),
                'invalid_samples': invalid_samples
            }
        }

class OptimizedCRNN:
    """Optimized CRNN model for ECI CAPTCHAs with 200x80 input"""
    
    def __init__(self, use_mixed_precision: bool = False):
        self.model = None
        self.prediction_model = None
        self.use_mixed_precision = use_mixed_precision
        
        if use_mixed_precision:
            # Enable mixed precision for faster training on newer GPUs
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
    
    def build_optimized_model(self) -> keras.Model:
        """Build optimized CRNN architecture for 200x80 images with enhanced stability"""
        logger.info("Building optimized CRNN model...")
        
        # Input layer for 200x80 images
        input_img = layers.Input(
            shape=(OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1), 
            name='image'
        )
        
        # Optimized CNN feature extractor for 200x80 input
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 40x100
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 20x50
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 10x25
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)  # 5x25 (preserve width more)
        x = layers.Dropout(0.2)(x)
        
        # Reshape for RNN (optimized for 200x80 -> 5x25 feature maps)
        new_shape = (25, 5 * 256)  # 25 time steps, 1280 features
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Optimized RNN layers with gradient clipping
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
        
        # Output layer for character sequence - one output per character position
        # We need 6 outputs (one per character in the CAPTCHA)
        # Each output should be a probability distribution over the 31 characters
        # 
        # The RNN output has shape (batch_size, 25, 128) where 25 is the sequence length
        # We need to map this to (batch_size, 6, 31) for the 6 characters with 31 classes each
        # 
        # Approach: Use a TimeDistributed Dense layer to map the RNN output to the character space,
        # then use a custom layer to select the appropriate time steps for each character position
        
        # First, we'll use a Dense layer to project to the right number of classes
        x = layers.Dense(OptimizedCaptchaConfig.NUM_CLASSES, name='character_projection')(x)  # No activation here, we'll apply it later
        
        # Now we need to reduce the sequence length from 25 to 6
        # We'll split the 25 time steps into 6 groups and take the average of each group
        # Group sizes: 4, 4, 4, 4, 4, 5 (to account for 25 total steps)
        group_sizes = [4, 4, 4, 4, 4, 5]
        char_outputs = []
        
        start_idx = 0
        for i, size in enumerate(group_sizes):
            # Extract the time steps for this character
            char_time_steps = layers.Lambda(lambda x: x[:, start_idx:start_idx+size, :])(x)
            # Average the time steps to get a single prediction per character
            char_output = layers.GlobalAveragePooling1D()(char_time_steps)
            # Apply softmax activation
            char_output = layers.Activation('softmax', name=f'char_{i}')(char_output)
            # Reshape to add back the character dimension
            char_output = layers.Reshape((1, OptimizedCaptchaConfig.NUM_CLASSES))(char_output)
            char_outputs.append(char_output)
            start_idx += size
        
        # Concatenate all character outputs
        outputs = layers.Concatenate(axis=1)(char_outputs)
        
        # Handle mixed precision output
        if self.use_mixed_precision:
            outputs = layers.Activation('softmax', dtype='float32', name='predictions')(outputs)
        
        model = keras.Model(input_img, outputs, name='OptimizedCRNN_200x80')
        logger.info(f"CRNN model built successfully with {model.count_params():,} parameters")
        return model
    
    def compile_model(self, learning_rate: float = None):
        """Compile model with optimized settings and enhanced stability"""
        if learning_rate is None:
            learning_rate = OptimizedCaptchaConfig.LEARNING_RATE
        
        logger.info(f"Compiling CRNN model with learning_rate={learning_rate}")
        self.model = self.build_optimized_model()
        
        # Log model input/output shapes for debugging
        logger.info(f"Model input shape: {self.model.input_shape}")
        logger.info(f"Model output shape: {self.model.output_shape}")
        
        # Use Adam optimizer with enhanced gradient clipping and stability settings
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,  # Gradient clipping to prevent exploding gradients
            # clipvalue=0.5,  # Removed to avoid conflict with clipnorm
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7  # Increased epsilon for numerical stability
        )
        
        # For mixed precision, loss scaling is handled automatically
        # Using sparse_categorical_crossentropy for sequence classification
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False  # Use graph execution for better performance
        )
        
        # Log model summary
        logger.debug("Model summary:")
        try:
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            for line in model_summary:
                logger.debug(line)
        except Exception as e:
            logger.debug(f"Could not print model summary: {e}")
        
        # Test model with dummy input to ensure it works
        try:
            dummy_input = np.random.random((1, OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1))
            dummy_output = self.model.predict(dummy_input, verbose=0)
            logger.info(f"Model test prediction shape: {dummy_output.shape}")
            logger.info(f"Model test prediction min: {np.min(dummy_output)}, max: {np.max(dummy_output)}")
            logger.info(f"Model test prediction has NaN: {np.isnan(dummy_output).any()}")
            logger.info(f"Model test prediction has Inf: {np.isinf(dummy_output).any()}")
            
            if np.isnan(dummy_output).any() or np.isinf(dummy_output).any():
                logger.error("Model produced invalid predictions during test!")
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
        
        logger.info(f"Model compiled with {self.model.count_params():,} parameters")
        return self.model

class EfficientSegmentationModel:
    """Efficient segmentation-based model optimized for 200x80 ECI patterns"""
    
    def __init__(self):
        self.model = None
        # Assuming text occupies only upto 80% width, and 50% height of captcha image
        self.character_width = OptimizedCaptchaConfig.IMG_WIDTH*0.8 // OptimizedCaptchaConfig.CAPTCHA_LENGTH
        self.character_height = OptimizedCaptchaConfig.IMG_HEIGHT*0.5
    
    def extract_character_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract character regions using improved vertical projection for 200x80 images"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Threshold to binary
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Vertical projection to find character boundaries
            vertical_projection = np.sum(binary, axis=0)
            
            # Smooth projection to reduce noise
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            smooth_projection = np.convolve(vertical_projection, kernel, mode='same')
            
            # Find character boundaries using improved method
            threshold = np.max(smooth_projection) * 0.15

            char_regions = []
            # First identify overall text region based on pixel projection density
            overall_text_region = np.where(smooth_projection > threshold*0.5)[0]
            
            # Define the search boundaries
            if overall_text_region.size > 0:
                text_start = overall_text_region[0]
                text_end = overall_text_region[-1]
            else:
                # If no text region found, use full width
                text_start = 0
                text_end = len(smooth_projection) - 1
            
            # Dynamic character boundary detection within text region only
            in_char = False
            char_starts = []
            char_ends = []
            
            # Only search within the identified text region
            for i in range(text_start, text_end + 1):
                val = smooth_projection[i]
                if val > threshold and not in_char:
                    char_starts.append(i)
                    in_char = True
                elif val <= threshold and in_char:
                    char_ends.append(i)
                    in_char = False
            
            if in_char:  # Handle case where last character goes to end of text region
                char_ends.append(text_end)
            
            # If we don't get exactly 6 regions, fall back to equal division within text region
            if len(char_starts) != OptimizedCaptchaConfig.CAPTCHA_LENGTH or len(char_ends) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.debug(f"Dynamic segmentation found {len(char_starts)} regions, using equal division within text region")
                
                # Calculate character width based on text region, not full image
                text_region_width = text_end - text_start
                char_width = text_region_width // OptimizedCaptchaConfig.CAPTCHA_LENGTH
                
                for i in range(OptimizedCaptchaConfig.CAPTCHA_LENGTH):
                    start_x = text_start + (i * char_width)
                    end_x = text_start + ((i + 1) * char_width)
                    
                    # Extract character region
                    char_region = gray[:, start_x:end_x]
                    
                    # Resize to standard size (maintain aspect ratio)
                    char_region = cv2.resize(char_region, (40, 64))  # Proportional to 200x80
                    
                    # Normalize
                    char_region = char_region.astype(np.float32) / 255.0
                    
                    char_regions.append(char_region)
            else:
                # Use dynamic boundaries
                for start, end in zip(char_starts, char_ends):
                    char_region = gray[:, start:end]
                    char_region = cv2.resize(char_region, (40, 64))
                    char_region = char_region.astype(np.float32) / 255.0
                    char_regions.append(char_region)
            
            return char_regions
            
        except Exception as e:
            logger.error(f"Character extraction failed: {e}")
            # Return empty regions as fallback
            return [np.zeros((64, 40), dtype=np.float32) for _ in range(OptimizedCaptchaConfig.CAPTCHA_LENGTH)]
    
    def build_character_classifier(self) -> keras.Model:
        """Build optimized individual character classifier for 40x64 character images"""
        model = keras.Sequential([
            layers.Input(shape=(64, 40, 1)),
            
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.1),
            
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(OptimizedCaptchaConfig.NUM_CLASSES, activation='softmax')
        ], name='CharacterClassifier_40x64')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=OptimizedCaptchaConfig.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class ProductionCaptchaSolver:
    """Production-ready CAPTCHA solver with ensemble and fallback for 200x80 images"""
    
    def __init__(self, model_dir: str = "models", fallback_api_key: str = None,
                 config_file: str = None, use_gpu: str = "auto", preprocessing_method: str = "light_smoothing"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.preprocessor = AdvancedPreprocessor(preprocessing_method=preprocessing_method)
        self.preprocessing_method = preprocessing_method  # Store preprocessing method for model loading
        self.crnn_model = None
        self.segmentation_model = None
        self.fallback_api_key = fallback_api_key
        
        # Setup GPU using new accelerator system
        self.gpu_available = False  # Default to CPU
        if use_gpu != "disable":
            try:
                if GPU_ACCELERATOR_AVAILABLE and get_accelerator is not None:
                    accelerator = get_accelerator()
                    gpu_info = accelerator.get_device_info()
                    if gpu_info["backend"] != "cpu":
                        self.gpu_available = True
                        logger.info(f"✓ GPU enabled: {gpu_info['backend']} on {gpu_info['device']}")
                    else:
                        if use_gpu == "force":
                            raise RuntimeError("GPU forced but not available")
                        logger.info("No GPU available, using CPU")
                else:
                    if use_gpu == "force":
                        raise RuntimeError("GPU forced but accelerator not available")
                    logger.info("No GPU available, using CPU")
            except Exception as e:
                if use_gpu == "force":
                    raise RuntimeError(f"GPU setup failed: {e}")
                logger.warning(f"GPU setup failed, using CPU: {e}")
                self.gpu_available = False
        
        # Performance tracking
        self.stats = {
            'total_attempts': 0,
            'ml_successes': 0,
            'api_fallbacks': 0,
            'failures': 0,
            'avg_inference_time': 0.0,
            'gpu_used': self.gpu_available
        }
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Load models if available
        self.load_models(preprocessing_method=self.preprocessing_method)
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "confidence_threshold": OptimizedCaptchaConfig.CONFIDENCE_THRESHOLD,
            "max_retries": 3,
            "use_ensemble": True,
            "save_solved_captchas": True
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        return default_config
    
    def load_models(self, preprocessing_method: str = "light_smoothing"):
        """Load trained models from disk"""
        try:
            # Enable unsafe deserialization for Lambda layers
            keras.config.enable_unsafe_deserialization()
            
            # Load CRNN model if it exists
            crnn_model_path = self.model_dir / f"optimized_crnn_200x80_{preprocessing_method}.keras"
            if crnn_model_path.exists():
                self.crnn_model = keras.models.load_model(crnn_model_path, safe_mode=False)
                logger.info(f"✓ CRNN model loaded: {crnn_model_path}")
            else:
                logger.warning(f"CRNN model not found: {crnn_model_path}")
            
            # Load segmentation model if it exists
            segmentation_model_path = self.model_dir / f"optimized_segmentation_200x80_{preprocessing_method}.keras"
            if segmentation_model_path.exists():
                self.segmentation_model = keras.models.load_model(segmentation_model_path, safe_mode=False)
                logger.info(f"✓ Segmentation model loaded: {segmentation_model_path}")
            else:
                logger.warning(f"Segmentation model not found: {segmentation_model_path}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Reset models to None if loading failed
            self.crnn_model = None
            self.segmentation_model = None
    
    def solve(self, image_path: str, confidence_threshold: float = None,
             save_to_dataset: bool = None) -> Optional[str]:
        """Solve CAPTCHA with optimized ensemble approach for 200x80 images"""
        if confidence_threshold is None:
            confidence_threshold = self.config.get("confidence_threshold", OptimizedCaptchaConfig.CONFIDENCE_THRESHOLD)
        
        if save_to_dataset is None:
            save_to_dataset = self.config.get("save_solved_captchas", True)
        
        start_time = time.time()
        self.stats['total_attempts'] += 1
        
        try:
            # Validate image first
            if not self._validate_image(image_path):
                logger.error(f"Invalid image: {image_path}")
                self._update_stats('failure', time.time() - start_time)
                return None
            
            # Primary: Segmentation approach (usually more reliable for this pattern)
            if self.segmentation_model:
                result = self._solve_with_segmentation(image_path, confidence_threshold)
                if result:
                    self._update_stats('ml_success', time.time() - start_time)
                    if save_to_dataset:
                        self._save_to_dataset(image_path, result)
                    return result
            
            # Secondary: CRNN approach
            if self.crnn_model:
                result = self._solve_with_crnn(image_path, confidence_threshold)
                if result:
                    self._update_stats('ml_success', time.time() - start_time)
                    if save_to_dataset:
                        self._save_to_dataset(image_path, result)
                    return result
            
            # Fallback: API service
            if self.fallback_api_key:
                result = self._solve_with_api(image_path)
                if result:
                    self._update_stats('api_fallback', time.time() - start_time)
                    if save_to_dataset:
                        self._save_to_dataset(image_path, result)
                    return result
            
            # Ultimate fallback: pattern-based OCR
            result = self._solve_with_ocr_fallback(image_path)
            if result:
                self._update_stats('ml_success', time.time() - start_time)
                if save_to_dataset:
                    self._save_to_dataset(image_path, result)
                return result
            
            self._update_stats('failure', time.time() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"Error solving CAPTCHA {image_path}: {e}")
            self._update_stats('failure', time.time() - start_time)
            return None
    
    def _validate_image(self, image_path: str) -> bool:
        """Validate image format and dimensions"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            height, width = img.shape[:2]
            
            # Accept 200x80 or allow reasonable variations
            if not (150 <= width <= 250 and 60 <= height <= 100):
                logger.warning(f"Unusual image dimensions: {width}x{height}")
            
            return True
            
        except Exception:
            return False
    
    def _solve_with_segmentation(self, image_path: str, confidence_threshold: float) -> Optional[str]:
        """Solve using segmentation approach for 200x80 images"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Ensure segmentation model is loaded
            if self.segmentation_model is None:
                # Try to load the model from disk if available
                model_path = self.model_dir / f"optimized_segmentation_200x80_{self.preprocessing_method}.keras"
                if model_path.exists():
                    from tensorflow import keras
                    self.segmentation_model = keras.models.load_model(model_path)
                    logger.info(f"Loaded segmentation model from {model_path}")
                else:
                    logger.warning("Segmentation model is not loaded and no saved model found.")
                    return None

            # Preprocess
            processed = self.preprocessor.preserve_green_text_during_line_removal(img)

            # Extract characters
            seg_model = EfficientSegmentationModel()
            char_regions = seg_model.extract_character_regions(processed)

            if len(char_regions) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.debug(f"Expected {OptimizedCaptchaConfig.CAPTCHA_LENGTH} characters, got {len(char_regions)}")
                return None

            # Predict each character
            predictions = []
            confidences = []

            for char_img in char_regions:
                char_input = np.expand_dims(np.expand_dims(char_img, axis=-1), axis=0)
                pred_probs = self.segmentation_model.predict(char_input, verbose=0)

                char_pred = np.argmax(pred_probs)
                char_conf = np.max(pred_probs)

                predictions.append(OptimizedCaptchaConfig.NUM_TO_CHAR[char_pred])
                confidences.append(char_conf)

            # Check overall confidence
            avg_confidence = np.mean(confidences)

            if avg_confidence >= confidence_threshold:
                result = ''.join(predictions)
                logger.debug(f"Segmentation solved: {result} (confidence: {avg_confidence:.3f})")
                return result

            return None

        except Exception as e:
            logger.warning(f"Segmentation solving failed: {e}")
            return None
    
    def _solve_with_crnn(self, image_path: str, confidence_threshold: float) -> Optional[str]:
        """Solve using CRNN approach with proper validation for 6-character outputs"""
        try:
            # Check if CRNN model is loaded
            if self.crnn_model is None:
                # Try to load the model from disk if available
                model_path = self.model_dir / f"optimized_crnn_200x80_{self.preprocessing_method}.keras"
                if model_path.exists():
                    from tensorflow import keras
                    self.crnn_model = keras.models.load_model(model_path)
                    logger.info(f"Loaded CRNN model from {model_path}")
                else:
                    logger.warning("CRNN model is not loaded and no saved model found.")
                    return None

            # Preprocess image
            processed_img = self.preprocessor.preprocess_image(image_path)
            input_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension

            # Predict with CRNN model
            predictions = self.crnn_model.predict(input_img, verbose=0)
            
            # Decode predictions
            predicted_chars = []
            avg_confidence = 0.0
            
            # For each character position (6 total)
            for i in range(OptimizedCaptchaConfig.CAPTCHA_LENGTH):
                # Get probabilities for this character position
                char_probs = predictions[0, i, :]
                # Get the predicted character and its confidence
                char_idx = np.argmax(char_probs)
                char_conf = np.max(char_probs)
                
                # Check if character is in our vocabulary
                if char_idx < len(OptimizedCaptchaConfig.NUM_TO_CHAR):
                    predicted_chars.append(OptimizedCaptchaConfig.NUM_TO_CHAR[char_idx])
                    avg_confidence += char_conf
                else:
                    # Invalid character prediction
                    logger.debug(f"CRNN predicted invalid character index: {char_idx}")
                    return None
            
            avg_confidence /= OptimizedCaptchaConfig.CAPTCHA_LENGTH
            result = ''.join(predicted_chars)
            
            # Validate that we have exactly 6 characters
            if len(result) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.debug(f"CRNN produced invalid length solution: {len(result)} characters")
                return None
                
            # Check confidence threshold
            if avg_confidence >= confidence_threshold:
                logger.debug(f"CRNN solved: {result} (confidence: {avg_confidence:.3f})")
                return result
            else:
                logger.debug(f"CRNN solution below confidence threshold: {result} ({avg_confidence:.3f})")
                return None
            
        except Exception as e:
            logger.warning(f"CRNN solving failed: {e}")
            return None
    
    def _solve_with_ocr_fallback(self, image_path: str) -> Optional[str]:
        """Fallback OCR using optimized preprocessing for 200x80 images"""
        try:
            import pytesseract
            
            # Aggressive preprocessing for OCR
            img = cv2.imread(image_path)
            processed = self.preprocessor.preprocess_image(image_path)
            
            # Convert back to uint8 for OCR and upscale for better recognition
            ocr_img = (processed[:, :, 0] * 255).astype(np.uint8)
            
            # Upscale for better OCR (200x80 -> 400x160)
            ocr_img = cv2.resize(ocr_img, (400, 160), interpolation=cv2.INTER_CUBIC)
            
            # OCR configuration for alphanumeric only
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            
            result = pytesseract.image_to_string(ocr_img, config=config).strip()
            
            if len(result) == OptimizedCaptchaConfig.CAPTCHA_LENGTH and result.isalnum():
                logger.debug(f"OCR fallback solved: {result}")
                return result.lower()  # Normalize to lowercase
            
            return None
            
        except ImportError:
            logger.warning("pytesseract not available for OCR fallback")
            return None
        except Exception as e:
            logger.warning(f"OCR fallback failed: {e}")
            return None
    
    def _solve_with_api(self, image_path: str) -> Optional[str]:
        """API fallback (implement with actual service)"""
        try:
            # This would integrate with AntiCaptcha or similar
            logger.debug("API fallback would be called here")
            return None
            
        except Exception as e:
            logger.warning(f"API fallback failed: {e}")
            return None
    
    def _save_to_dataset(self, image_path: str, solution: str):
        """Save solved CAPTCHA to dataset for continuous learning, only if solution is valid"""
        try:
            # Validate that solution is exactly 6 characters long
            if len(solution) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.debug(f"Skipping save to dataset: solution '{solution}' is not {OptimizedCaptchaConfig.CAPTCHA_LENGTH} characters long")
                return
                
            # Validate that solution contains only valid characters
            valid_chars = set(OptimizedCaptchaConfig.CHARACTERS)
            if not all(c in valid_chars for c in solution):
                logger.debug(f"Skipping save to dataset: solution '{solution}' contains invalid characters")
                return
            
            dataset_dir = Path("captcha_dataset")
            dataset_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = f"{solution.upper()}_{timestamp}_PRODUCTION"
            
            # Copy image
            import shutil
            img_dest = dataset_dir / f"{base_name}.png"
            shutil.copy2(image_path, img_dest)
            
            # Create JSON metadata
            metadata = {
                "solution": solution,
                "timestamp": datetime.now().isoformat(),
                "source": "production_solver",
                "original_path": str(image_path),
                "confidence": "high",
                "image_dimensions": "200x80"
            }
            
            json_dest = dataset_dir / f"{base_name}.json"
            with open(json_dest, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved to dataset: {base_name}")
            
        except Exception as e:
            logger.warning(f"Failed to save to dataset: {e}")
    
    def _update_stats(self, result_type: str, inference_time: float):
        """Update performance statistics"""
        if result_type == 'ml_success':
            self.stats['ml_successes'] += 1
        elif result_type == 'api_fallback':
            self.stats['api_fallbacks'] += 1
        elif result_type == 'failure':
            self.stats['failures'] += 1
        
        # Update average inference time
        total_times = self.stats['total_attempts']
        self.stats['avg_inference_time'] = (
            (self.stats['avg_inference_time'] * (total_times - 1) + inference_time) / total_times
        )
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        if self.stats['total_attempts'] == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats['ml_success_rate'] = self.stats['ml_successes'] / self.stats['total_attempts']
        stats['api_usage_rate'] = self.stats['api_fallbacks'] / self.stats['total_attempts']
        stats['failure_rate'] = self.stats['failures'] / self.stats['total_attempts']
        
        # Add GPU memory info if available
        try:
            if GPU_ACCELERATOR_AVAILABLE and get_gpu_memory_info is not None:
                gpu_memory = get_gpu_memory_info()
                if gpu_memory:
                    stats['gpu_memory'] = gpu_memory
        except Exception as e:
            logger.debug(f"GPU memory info unavailable: {e}")
        
        return stats
    
    def train_models(self, datasets: Dict):
        """Train both CRNN and segmentation models with GPU support"""
        logger.info("Starting optimized model training for 200x80 images...")
        
        # Setup GPU for training using new accelerator system
        if OptimizedCaptchaConfig.USE_GPU_FOR_TRAINING:
            try:
                if GPU_ACCELERATOR_AVAILABLE and get_accelerator is not None:
                    accelerator = get_accelerator()
                    gpu_info = accelerator.get_device_info()
                    if gpu_info["backend"] != "cpu":
                        logger.info(f"✓ Training with GPU acceleration: {gpu_info['backend']}")
                        # Setup TensorFlow GPU if available
                        setup_tensorflow_gpu()
                    else:
                        logger.info("No GPU available - training with CPU")
                else:
                    logger.info("GPU accelerator not available - training with CPU")
            except Exception as e:
                logger.warning(f"GPU setup for training failed: {e}")
        
        # Debug: Print dataset split stats
        logger.info(f"DEBUG: Dataset stats: {datasets['stats']}")
        train_samples = datasets['stats'].get('train_samples', 0)
        logger.info(f"DEBUG: Train samples count: {train_samples}")
        logger.debug(f"DEBUG: Full train paths (first 10): {datasets['train'][0][:10]} ... total: {len(datasets['train'][0])}")
        logger.debug(f"DEBUG: Full train labels (first 10): {datasets['train'][1][:10]} ... total: {len(datasets['train'][1])}")
        logger.info("DEBUG: Entering segmentation training block.")
        self._train_segmentation(datasets)
        logger.info("DEBUG: Segmentation training block exited.")
        if train_samples > 1000:
            logger.info("DEBUG: Entering CRNN training block.")
            self._train_crnn(datasets)
            logger.info("DEBUG: CRNN training block exited.")
        else:
            logger.warning(f"DEBUG: Skipping CRNN training, train_samples={train_samples}")
        logger.info("DEBUG: Entering model saving block.")
        self.save_models(preprocessing_method=self.preprocessor.preprocessing_method)
        logger.info("DEBUG: Model saving block exited.")
        logger.info("Training completed successfully!")
    
    def _train_segmentation(self, datasets: Dict):
        try:
            logger.debug("DEBUG: Segmentation training started.")
            logger.info("Training segmentation model for 200x80 images...")
            char_images = []
            char_labels = []
            seg_model = EfficientSegmentationModel()
            train_paths, train_labels = datasets['train']
            max_training_samples = min(1000, len(train_paths))
            logger.debug(f"DEBUG: max_training_samples for segmentation: {max_training_samples}")
            for i, (img_path, label) in enumerate(zip(train_paths[:max_training_samples], train_labels[:max_training_samples])):
                if i % 200 == 0:
                    logger.info(f"Processing training sample {i}/{max_training_samples}")
                logger.debug(f"DEBUG: Processing img_path: {img_path}, label: {label}")
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        processed = self.preprocessor.preserve_green_text_during_line_removal(img)
                        char_regions = seg_model.extract_character_regions(processed)
                        if len(char_regions) == len(label):
                            for char_img, char_label in zip(char_regions, label):
                                char_images.append(np.expand_dims(char_img, axis=-1))
                                char_labels.append(OptimizedCaptchaConfig.CHAR_TO_NUM[char_label])
                        else:
                            logger.warning(f"DEBUG: Char region mismatch for {img_path}: expected {len(label)}, got {len(char_regions)}")
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
            logger.debug(f"DEBUG: Total char_images collected: {len(char_images)}")
            logger.debug(f"DEBUG: Total char_labels collected: {len(char_labels)}")
            if len(char_images) > 500:
                char_images = np.array(char_images)
                char_labels = np.array(char_labels)
                logger.info(f"Training segmentation model with {len(char_images)} character samples")
                self.segmentation_model = seg_model.build_character_classifier()
                
                # Enhanced callbacks with better monitoring
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
                    keras.callbacks.ModelCheckpoint(
                        str(self.model_dir / "best_segmentation.h5"),
                        save_best_only=True,
                        monitor='val_accuracy'
                    ),
                    # Custom callback for detailed metrics logging
                    TrainingMetricsLogger()
                ]
                
                history = self.segmentation_model.fit(
                    char_images, char_labels,
                    batch_size=OptimizedCaptchaConfig.BATCH_SIZE,
                    epochs=40,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                final_accuracy = max(history.history['val_accuracy'])
                logger.info(f"Segmentation model trained successfully. Best validation accuracy: {final_accuracy:.3f}")
                logger.debug(f"DEBUG: Segmentation training history: {history.history}")
                
                # Save detailed training metrics
                self._save_training_metrics(history.history, "segmentation")
            else:
                logger.warning("Insufficient character data for segmentation training")
                logger.debug(f"DEBUG: char_images count: {len(char_images)}")
        except Exception as e:
            logger.error(f"Segmentation training failed: {e}")
    
    def _train_crnn(self, datasets: Dict):
        try:
            logger.debug("DEBUG: CRNN training started.")
            logger.info("Training CRNN model with enhanced debugging...")
            
            # Create CRNN model with additional debugging
            crnn = OptimizedCRNN()
            self.crnn_model = crnn.compile_model()
            
            # Get datasets
            train_dataset = datasets['train_dataset']
            val_dataset = datasets['val_dataset']
            
            # Enhanced debugging information
            logger.info(f"DEBUG: train_dataset type: {type(train_dataset)}")
            logger.info(f"DEBUG: val_dataset type: {type(val_dataset)}")
            logger.debug(f"DEBUG: train_dataset: {train_dataset}")
            logger.debug(f"DEBUG: val_dataset: {val_dataset}")
            
            # Add data validation before training
            logger.info("Validating training data...")
            self._validate_training_data(train_dataset, val_dataset)
            
            # Enhanced callbacks with better monitoring
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10, 
                    restore_best_weights=True,
                    monitor='val_loss',
                    mode='min'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, 
                    patience=5,
                    monitor='val_loss',
                    mode='min',
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    str(self.model_dir / "best_crnn.keras"),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min'
                ),
                # Custom callback for NaN detection
                NanDetectionCallback(),
                # Custom callback for detailed metrics logging
                TrainingMetricsLogger()
            ]
            
            # Train with enhanced error handling
            logger.info("Starting CRNN training...")
            history = self.crnn_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=OptimizedCaptchaConfig.EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            # Log training results
            final_accuracy = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0
            final_loss = min(history.history['val_loss']) if 'val_loss' in history.history else float('inf')
            logger.info(f"CRNN model trained successfully. Best validation accuracy: {final_accuracy:.3f}, Best validation loss: {final_loss:.3f}")
            logger.debug(f"DEBUG: CRNN training history: {history.history}")
            
            # Save detailed training metrics
            self._save_training_metrics(history.history, "crnn")
            
        except Exception as e:
            logger.error(f"DEBUG: CRNN training failed: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            self.crnn_model = None
            logger.error(f"CRNN training failed: {e}")
            self.crnn_model = None
    
    def _validate_training_data(self, train_dataset, val_dataset):
        """Validate training data to detect potential issues causing NaN losses"""
        try:
            logger.info("Starting data validation...")
            
            # Check a few samples from training dataset
            sample_count = 0
            for batch in train_dataset.take(3):  # Check first 3 batches
                images, labels = batch
                sample_count += len(images)
                
                # Check for NaN or infinite values in images
                if tf.reduce_any(tf.math.is_nan(images)):
                    logger.error(f"NaN values found in training images in batch")
                    return False
                    
                if tf.reduce_any(tf.math.is_inf(images)):
                    logger.error(f"Infinite values found in training images in batch")
                    return False
                    
                # Check label validity
                if tf.reduce_any(labels < 0) or tf.reduce_any(labels >= OptimizedCaptchaConfig.NUM_CLASSES):
                    logger.error(f"Invalid label values found in training labels: min={tf.reduce_min(labels)}, max={tf.reduce_max(labels)}")
                    return False
                    
                # Check label shape
                if labels.shape[1] != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                    logger.error(f"Invalid label shape: expected {OptimizedCaptchaConfig.CAPTCHA_LENGTH}, got {labels.shape[1]}")
                    return False
                    
                logger.debug(f"Batch {sample_count//len(images)}: Images shape={images.shape}, Labels shape={labels.shape}")
            
            logger.info(f"Data validation passed for {sample_count} training samples")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def save_models(self, preprocessing_method=None):
        try:
            logger.debug("DEBUG: Model saving started.")
            suffix = f"_{preprocessing_method}" if preprocessing_method else ""
            if self.crnn_model:
                model_path = self.model_dir / f"optimized_crnn_200x80{suffix}.keras"
                self.crnn_model.save(model_path)
                logger.info(f"✓ CRNN model saved: {model_path}")
            else:
                logger.warning("DEBUG: CRNN model not present, not saved.")
            if self.segmentation_model:
                model_path = self.model_dir / f"optimized_segmentation_200x80{suffix}.keras"
                self.segmentation_model.save(model_path)
                logger.info(f"✓ Segmentation model saved: {model_path}")
            else:
                logger.warning("DEBUG: Segmentation model not present, not saved.")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            
    def _save_training_metrics(self, history: Dict, model_type: str):
        """Save detailed training metrics to JSON and CSV files for analysis"""
        try:
            import json
            import pandas as pd
            from datetime import datetime
            
            # Create metrics directory if it doesn't exist
            metrics_dir = self.model_dir / "training_metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            # Prepare metrics data
            metrics_data = {
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
                "epochs": len(history.get("loss", [])),
                "final_metrics": {},
                "best_metrics": {},
                "history": history
            }
            
            # Add final metrics
            if "loss" in history:
                metrics_data["final_metrics"]["loss"] = history["loss"][-1]
                metrics_data["best_metrics"]["loss"] = min(history["loss"])
                
            if "accuracy" in history:
                metrics_data["final_metrics"]["accuracy"] = history["accuracy"][-1]
                metrics_data["best_metrics"]["accuracy"] = max(history["accuracy"])
                
            if "val_loss" in history:
                metrics_data["final_metrics"]["val_loss"] = history["val_loss"][-1]
                metrics_data["best_metrics"]["val_loss"] = min(history["val_loss"])
                
            if "val_accuracy" in history:
                metrics_data["final_metrics"]["val_accuracy"] = history["val_accuracy"][-1]
                metrics_data["best_metrics"]["val_accuracy"] = max(history["val_accuracy"])
            
            # Save to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = metrics_dir / f"{model_type}_training_metrics_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"Training metrics saved to: {json_file}")
            
            # Save to CSV for easier analysis
            if "epoch" not in history:
                # Add epoch numbers if not present
                epochs = list(range(1, len(history.get("loss", [])) + 1))
                csv_data = {"epoch": epochs}
            else:
                csv_data = {"epoch": history["epoch"]}
                
            # Add all available metrics
            for key, values in history.items():
                if key != "epoch":
                    csv_data[key] = values
                    
            df = pd.DataFrame(csv_data)
            csv_file = metrics_dir / f"{model_type}_training_metrics_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Training metrics CSV saved to: {csv_file}")
            
        except Exception as e:
            logger.error(f"Error saving training metrics: {e}")
            
    def _create_training_summary(self, datasets: Dict):
        """Create a comprehensive training summary with all metrics and statistics"""
        try:
            import json
            from datetime import datetime
            
            # Create summary directory if it doesn't exist
            summary_dir = self.model_dir / "training_summaries"
            summary_dir.mkdir(exist_ok=True)
            
            # Collect training summary data
            summary_data = {
                "training_completed": datetime.now().isoformat(),
                "model_types_trained": [],
                "dataset_stats": datasets.get('stats', {}),
                "model_details": {},
                "training_metrics": {}
            }
            
            # Add model information
            if self.crnn_model:
                summary_data["model_types_trained"].append("crnn")
                summary_data["model_details"]["crnn"] = {
                    "input_shape": str(self.crnn_model.input_shape) if hasattr(self.crnn_model, 'input_shape') else "Unknown",
                    "output_shape": str(self.crnn_model.output_shape) if hasattr(self.crnn_model, 'output_shape') else "Unknown",
                    "parameters": int(self.crnn_model.count_params()) if hasattr(self.crnn_model, 'count_params') else "Unknown"
                }
                
            if self.segmentation_model:
                summary_data["model_types_trained"].append("segmentation")
                summary_data["model_details"]["segmentation"] = {
                    "input_shape": str(self.segmentation_model.input_shape) if hasattr(self.segmentation_model, 'input_shape') else "Unknown",
                    "output_shape": str(self.segmentation_model.output_shape) if hasattr(self.segmentation_model, 'output_shape') else "Unknown",
                    "parameters": int(self.segmentation_model.count_params()) if hasattr(self.segmentation_model, 'count_params') else "Unknown"
                }
            
            # Add system information
            summary_data["system_info"] = {
                "gpu_used": self.gpu_available,
                "preprocessing_method": self.preprocessor.preprocessing_method
            }
            
            # Try to get GPU information
            try:
                if GPU_ACCELERATOR_AVAILABLE and get_gpu_memory_info is not None:
                    gpu_info = get_gpu_memory_info()
                    summary_data["system_info"]["gpu_info"] = gpu_info
            except Exception as e:
                logger.debug(f"Could not get GPU info: {e}")
            
            # Save training summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = summary_dir / f"training_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            logger.info(f"Training summary saved: {summary_file}")
            
            # Also save as a readable text file
            txt_file = summary_dir / f"training_summary_{timestamp}.txt"
            with open(txt_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("ECI CAPTCHA Solver Training Summary\n")
                f.write("=" * 60 + "\n")
                f.write(f"Training completed: {summary_data['training_completed']}\n")
                f.write(f"Models trained: {', '.join(summary_data['model_types_trained'])}\n")
                f.write(f"Preprocessing method: {summary_data['system_info']['preprocessing_method']}\n")
                f.write(f"GPU used: {summary_data['system_info']['gpu_used']}\n")
                f.write("\nDataset Statistics:\n")
                for key, value in summary_data['dataset_stats'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\nModel Details:\n")
                for model_type, details in summary_data['model_details'].items():
                    f.write(f"  {model_type.upper()} Model:\n")
                    for detail_key, detail_value in details.items():
                        f.write(f"    {detail_key}: {detail_value}\n")
                f.write("=" * 60 + "\n")
            logger.info(f"Training summary (text) saved: {txt_file}")
            
        except Exception as e:
            logger.error(f"Error creating training summary: {e}")

def create_test_visualization(test_results: List[Dict], output_dir: Path, accuracy: float):
    """Create visualization of test results"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'CAPTCHA Test Results - Accuracy: {accuracy:.3f}', fontsize=16)
        
        # 1. Accuracy pie chart
        correct_count = sum(1 for r in test_results if r['correct'])
        incorrect_count = len(test_results) - correct_count
        
        ax1.pie([correct_count, incorrect_count], 
                labels=[f'Correct ({correct_count})', f'Incorrect ({incorrect_count})'],
                colors=['green', 'red'], autopct='%1.1f%%')
        ax1.set_title('Overall Accuracy')
        
        # 2. Character-wise accuracy
        char_accuracy = {}
        for result in test_results:
            true_label = result['true_label']
            prediction = result['prediction']
            
            for i, char in enumerate(true_label):
                if char not in char_accuracy:
                    char_accuracy[char] = {'correct': 0, 'total': 0}
                char_accuracy[char]['total'] += 1
                
                if (prediction != 'FAILED' and 
                    len(prediction) > i and 
                    prediction[i].lower() == char.lower()):
                    char_accuracy[char]['correct'] += 1
        
        chars = list(char_accuracy.keys())
        accuracies = [char_accuracy[c]['correct'] / char_accuracy[c]['total'] 
                     for c in chars if char_accuracy[c]['total'] > 0]
        
        if chars and accuracies:
            ax2.bar(chars, accuracies)
            ax2.set_title('Character-wise Accuracy')
            ax2.set_xlabel('Characters')
            ax2.set_ylabel('Accuracy')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Sample results visualization
        sample_size = min(20, len(test_results))
        sample_results = test_results[:sample_size]
        
        y_pos = range(len(sample_results))
        colors = ['green' if r['correct'] else 'red' for r in sample_results]
        
        ax3.barh(y_pos, [1] * len(sample_results), color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{r['true_label']} → {r['prediction']}" for r in sample_results])
        ax3.set_title(f'Sample Results (First {sample_size})')
        ax3.set_xlabel('Result')
        
        # 4. Performance distribution
        correct_indices = [i for i, r in enumerate(test_results) if r['correct']]
        incorrect_indices = [i for i, r in enumerate(test_results) if not r['correct']]
        
        ax4.hist([correct_indices, incorrect_indices], 
                bins=20, label=['Correct', 'Incorrect'], 
                color=['green', 'red'], alpha=0.7)
        ax4.set_title('Performance Distribution Over Test Set')
        ax4.set_xlabel('Test Index')
        ax4.set_ylabel('Count')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = output_dir / "test_results_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed character analysis
        char_analysis_file = output_dir / "character_analysis.json"
        with open(char_analysis_file, 'w') as f:
            json.dump(char_accuracy, f, indent=2)
        
        logger.info(f"Test visualization saved to: {viz_file}")
        
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")

def debug_training_data(dataset_dir: str, csv_file: str = None):
    """Debug utility to analyze training data and identify potential issues"""
    logger.info("Starting training data debug analysis...")
    
    try:
        # Load data using the existing data loader
        data_loader = OptimizedDataLoader(dataset_dir, csv_file)
        image_paths, solutions = data_loader.load_from_csv()
        
        logger.info(f"Total samples loaded: {len(image_paths)}")
        
        # Analyze data statistics
        char_frequency = {}
        invalid_chars = set()
        
        for solution in solutions:
            # Check solution length
            if len(solution) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.warning(f"Invalid length solution: {solution} (length: {len(solution)})")
                continue
                
            # Analyze character frequency
            for char in solution:
                if char in OptimizedCaptchaConfig.CHARACTERS:
                    char_frequency[char] = char_frequency.get(char, 0) + 1
                else:
                    invalid_chars.add(char)
                    logger.warning(f"Invalid character found: {char} in solution {solution}")
        
        # Log character frequency analysis
        logger.info("Character frequency analysis:")
        sorted_chars = sorted(char_frequency.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars:
            percentage = (count / len(solutions)) * 100
            logger.info(f"  {char}: {count} ({percentage:.2f}%)")
        
        if invalid_chars:
            logger.error(f"Invalid characters found: {invalid_chars}")
        
        # Check a few sample images
        logger.info("Checking sample images...")
        sample_count = min(10, len(image_paths))
        for i in range(sample_count):
            try:
                img = cv2.imread(image_paths[i])
                if img is None:
                    logger.warning(f"Could not load image: {image_paths[i]}")
                else:
                    logger.debug(f"Sample {i+1}: {image_paths[i]} - shape: {img.shape}, solution: {solutions[i]}")
            except Exception as e:
                logger.error(f"Error loading sample image {image_paths[i]}: {e}")
        
        logger.info("Training data debug analysis completed.")
        return True
        
    except Exception as e:
        logger.error(f"Error in debug_training_data: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False

def test_preprocessing(image_path: str, preprocessing_method: str = "light_smoothing"):
    """Test preprocessing on a single image"""
    logger.info(f"Testing preprocessing on image: {image_path}")
    logger.info(f"Preprocessing method: {preprocessing_method}")
    
    try:
        preprocessor = AdvancedPreprocessor(preprocessing_method=preprocessing_method)
        result = preprocessor.preprocess_image(image_path)
        
        logger.info(f"Preprocessing result shape: {result.shape}")
        logger.info(f"Preprocessing result dtype: {result.dtype}")
        logger.info(f"Preprocessing result min: {np.min(result)}, max: {np.max(result)}")
        logger.info(f"Preprocessing result has NaN: {np.isnan(result).any()}")
        logger.info(f"Preprocessing result has Inf: {np.isinf(result).any()}")
        
        if np.isnan(result).any() or np.isinf(result).any():
            logger.error("Preprocessing produced invalid values!")
            return False
            
        logger.info("Preprocessing test completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error in test_preprocessing: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False

def test_model_loading(model_dir: str, preprocessing_method: str = "light_smoothing"):
    """Test model loading"""
    logger.info(f"Testing model loading from: {model_dir}")
    logger.info(f"Preprocessing method: {preprocessing_method}")
    
    try:
        solver = ProductionCaptchaSolver(model_dir, preprocessing_method=preprocessing_method)
        
        # Try to load models
        solver.load_models(preprocessing_method=preprocessing_method)
        
        if solver.crnn_model:
            logger.info("CRNN model loaded successfully")
            logger.info(f"CRNN model input shape: {solver.crnn_model.input_shape}")
            logger.info(f"CRNN model output shape: {solver.crnn_model.output_shape}")
        else:
            logger.warning("CRNN model not loaded")
            
        if solver.segmentation_model:
            logger.info("Segmentation model loaded successfully")
        else:
            logger.warning("Segmentation model not loaded")
            
        logger.info("Model loading test completed.")
        return True
        
    except Exception as e:
        logger.error(f"Error in test_model_loading: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False


def main():
    """Main execution function for 200x80 CAPTCHA solver"""
    parser = argparse.ArgumentParser(description="ECI CAPTCHA Solver for 200x80 Images")
    parser.add_argument("--dataset-dir", help="CAPTCHA dataset directory (required for train/test modes)")
    parser.add_argument("--csv-file", help="Consolidated CSV file path")
    parser.add_argument("--mode", choices=["train", "test", "solve", "debug-data", "test-preprocessing", "test-model-loading"], default="train")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--image-path", help="Single image path or directory containing PNG files (for solve mode)")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--use-gpu", choices=["auto", "force", "disable"], default="auto")
    parser.add_argument("--config-file", help="Configuration file path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--save-preprocessed", action="store_true", help="Save preprocessed images during training")
    parser.add_argument("--output-dir", default="output", help="Output directory for results and preprocessed images")
    parser.add_argument("--preprocessing-method", choices=["grayscale", "light_smoothing", "precise_mathematical"], 
                        default="light_smoothing", help="Preprocessing method to use")
    parser.add_argument("--solution-method", choices=["segment", "crnn", "ocr"], help="Force solution method: segment, crnn, or ocr")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ["train", "test", "debug-data"] and not args.dataset_dir:
        parser.error(f"--dataset-dir is required for {args.mode} mode")
    
    if args.mode in ["solve", "test-preprocessing"] and not args.image_path:
        parser.error(f"--image-path is required for {args.mode} mode")
    
    # Print GPU status if available
    if GPU_ACCELERATOR_AVAILABLE and print_gpu_status is not None:
        try:
            print_gpu_status()
        except Exception as e:
            logger.info(f"GPU status check failed: {e}")
    
    # Update configuration
    OptimizedCaptchaConfig.EPOCHS = args.epochs
    OptimizedCaptchaConfig.BATCH_SIZE = args.batch_size
    
    if args.mode == "debug-data":
        # Debug data mode
        logger.info("Running data debug analysis...")
        debug_training_data(args.dataset_dir, args.csv_file)
        return
    
    if args.mode == "test-preprocessing":
        # Test preprocessing mode
        logger.info("Running preprocessing test...")
        test_preprocessing(args.image_path, args.preprocessing_method)
        return
    
    if args.mode == "test-model-loading":
        # Test model loading mode
        logger.info("Running model loading test...")
        test_model_loading(args.model_dir, args.preprocessing_method)
        return
    
    if args.mode == "train":
        # Training mode
        logger.info("Starting optimized CAPTCHA solver training for 200x80 images...")
        
        # Setup GPU for training
        if args.use_gpu in ["auto", "force"]:
            # Setup GPU using the new accelerator system
            try:
                if GPU_ACCELERATOR_AVAILABLE and get_accelerator is not None:
                    accelerator = get_accelerator()
                    gpu_info = accelerator.get_device_info()
                    if gpu_info["backend"] != "cpu":
                        logger.info(f"✓ GPU available: {gpu_info['backend']} on {gpu_info['device']}")
                    else:
                        logger.warning("GPU requested but only CPU available")
                        if args.use_gpu == "force":
                            logger.error("GPU forced but not available - exiting")
                            return
                else:
                    logger.warning("GPU accelerator not available")
                    if args.use_gpu == "force":
                        logger.error("GPU forced but accelerator not available - exiting")
                        return
            except Exception as e:
                logger.error(f"GPU setup error: {e}")
                if args.use_gpu == "force":
                    return
        
        data_loader = OptimizedDataLoader(args.dataset_dir, args.csv_file, use_gpu=True, 
                                        preprocessing_method=args.preprocessing_method)
        datasets = data_loader.prepare_datasets()
        
        # Log dataset statistics
        logger.info(f"Dataset statistics: {datasets['stats']}")
        
        solver = ProductionCaptchaSolver(args.model_dir, config_file=args.config_file, 
                                       use_gpu=args.use_gpu, preprocessing_method=args.preprocessing_method)
        
        # Setup output directory for preprocessed images if requested
        if args.save_preprocessed:
            preprocessed_dir = Path("output") / "preprocessed_captchas"
            preprocessed_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Preprocessed images will be saved to: {preprocessed_dir}")
            
            # Save a few sample preprocessed images during training
            train_paths, _ = datasets['train']
            sample_paths = train_paths[:min(50, len(train_paths))]  # Save first 50 as samples
            
            logger.info(f"Saving {len(sample_paths)} sample preprocessed images...")
            for img_path in sample_paths:
                try:
                    _ = solver.preprocessor.preprocess_image(img_path, save_preprocessed=True, output_dir=str(preprocessed_dir))
                except Exception as e:
                    logger.debug(f"Failed to save preprocessed image for {img_path}: {e}")
        
        solver.train_models(datasets)
        
        logger.info("Training completed!")
        
    elif args.mode == "test":
        # Testing mode
        logger.info("Testing CAPTCHA solver for 200x80 images...")
        
        data_loader = OptimizedDataLoader(args.dataset_dir, args.csv_file, 
                                        preprocessing_method=args.preprocessing_method)
        datasets = data_loader.prepare_datasets()
        
        solver = ProductionCaptchaSolver(args.model_dir, config_file=args.config_file, 
                                       use_gpu=args.use_gpu, preprocessing_method=args.preprocessing_method)
        
        # Test on subset
        test_paths, test_labels = datasets['test']
        test_subset = min(500, len(test_paths))  # Test first 500
        
        # Setup output directory for test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_output_dir = Path(args.output_dir) / f"test_results_{timestamp}"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track test results for analysis
        test_results = []
        correct = 0
        total = 0
        
        logger.info(f"Testing on {test_subset} samples...")
        logger.info(f"Test results will be saved to: {test_output_dir}")
        
        for i, (img_path, true_label) in enumerate(zip(test_paths[:test_subset], test_labels[:test_subset])):
            prediction = solver.solve(img_path, args.confidence_threshold)
            
            is_correct = prediction and prediction.lower() == true_label.lower()
            if is_correct:
                correct += 1
            total += 1
            
            # Store result for analysis
            test_results.append({
                'image_path': img_path,
                'true_label': true_label,
                'prediction': prediction or 'FAILED',
                'correct': is_correct,
                'index': i
            })
            
            if (i + 1) % 50 == 0:
                logger.info(f"Tested {i+1}/{test_subset}, accuracy: {correct/total:.3f}")
        
        # Final results
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Final accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Save test results
        results_file = test_output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tested': total,
                    'correct_predictions': correct,
                    'accuracy': accuracy,
                    'test_subset_size': test_subset,
                    'timestamp': timestamp
                },
                'details': test_results
            }, f, indent=2)
        
        # Create visualization of results
        try:
            create_test_visualization(test_results, test_output_dir, accuracy)
            logger.info(f"Test visualization saved to: {test_output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create test visualization: {e}")
        
        stats = solver.get_performance_stats()
        logger.info(f"Performance stats: {stats}")
        
        # Save performance stats
        stats_file = test_output_dir / "performance_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Test completed! Results saved to: {test_output_dir}")
        
    elif args.mode == "solve":
        # Solve mode - handle single file or directory
        if not args.image_path:
            logger.error("Image path required for solve mode")
            return
        
        solver = ProductionCaptchaSolver(args.model_dir, config_file=args.config_file, 
                                       use_gpu=args.use_gpu, preprocessing_method=args.preprocessing_method)
        image_path = Path(args.image_path)
        
        def solve_with_method(solver, img_path, confidence_threshold, solution_method):
            method_used = None
            solution = None
            if solution_method == "segment":
                solution = solver._solve_with_segmentation(str(img_path), confidence_threshold)
                method_used = "segment"
            elif solution_method == "crnn":
                solution = solver._solve_with_crnn(str(img_path), confidence_threshold)
                method_used = "crnn"
            elif solution_method == "ocr":
                solution = solver._solve_with_ocr_fallback(str(img_path))
                method_used = "ocr"
            else:
                # Default fallback sequence
                if solver.segmentation_model:
                    solution = solver._solve_with_segmentation(str(img_path), confidence_threshold)
                    method_used = "segment" if solution else None
                if not solution and solver.crnn_model:
                    solution = solver._solve_with_crnn(str(img_path), confidence_threshold)
                    method_used = "crnn" if solution else method_used
                if not solution and solver.fallback_api_key:
                    solution = solver._solve_with_api(str(img_path))
                    method_used = "api" if solution else method_used
                if not solution:
                    solution = solver._solve_with_ocr_fallback(str(img_path))
                    method_used = "ocr" if solution else method_used
            return solution, method_used
        
        if image_path.is_file():
            # Single file mode
            start_time = time.time()
            solution, method_used = solve_with_method(solver, image_path, args.confidence_threshold, args.solution_method)
            solve_time = time.time() - start_time
            
            # Save solution to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            solution_dir = Path("output") / "captcha_solutions"
            solution_dir.mkdir(parents=True, exist_ok=True)
            
            # Create solution record
            solution_data = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'solution': solution or 'FAILED',
                'solve_time': solve_time,
                'success': solution is not None,
                'confidence_threshold': args.confidence_threshold,
                'timestamp': timestamp,
                'method_used': method_used or 'fallback_sequence'
            }
            
            # Save solution JSON
            solution_file = solution_dir / f"{image_path.stem}_solution_{timestamp}.json"
            with open(solution_file, 'w') as f:
                json.dump(solution_data, f, indent=2)
            
            # Copy original image to solutions directory
            import shutil
            original_copy = solution_dir / f"{image_path.stem}_original.png"
            shutil.copy2(image_path, original_copy)
            
            # Save preprocessed image
            try:
                preprocessed_dir = solution_dir / "preprocessed"
                preprocessed_dir.mkdir(exist_ok=True)
                _ = solver.preprocessor.preprocess_image(str(image_path), save_preprocessed=True, output_dir=str(preprocessed_dir))
                logger.info(f"Saved preprocessed image to: {preprocessed_dir}")
            except Exception as e:
                logger.debug(f"Failed to save preprocessed image: {e}")
            
            if solution:
                print(f"CAPTCHA solution: {solution} (solved in {solve_time:.3f}s)")
                print(f"Method used: {method_used}")
                print(f"Solution saved to: {solution_file}")
            else:
                print(f"Failed to solve CAPTCHA (attempted for {solve_time:.3f}s)")
                print(f"Method used: {method_used or 'fallback_sequence'}")
            
            print(f"Performance stats: {solver.get_performance_stats()}")
            
        elif image_path.is_dir():
            # Directory mode - solve all PNG files
            png_files = list(image_path.glob("*.png"))
            if not png_files:
                print(f"No PNG files found in directory: {image_path}")
                return
            
            print(f"Found {len(png_files)} PNG files to solve...")
            
            # Setup output directory for batch solve results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_solution_dir = Path("output") / "captcha_solutions" / f"batch_{timestamp}"
            batch_solution_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            originals_dir = batch_solution_dir / "originals"
            preprocessed_dir = batch_solution_dir / "preprocessed"
            originals_dir.mkdir(exist_ok=True)
            preprocessed_dir.mkdir(exist_ok=True)
            
            # Track batch solve results
            batch_results = []
            successful_solves = 0
            total_time = 0
            
            import shutil
            
            for i, png_file in enumerate(png_files):
                file_start_time = time.time()
                solution, method_used = solve_with_method(solver, png_file, args.confidence_threshold, args.solution_method)
                solve_time = time.time() - file_start_time
                total_time += solve_time
                
                is_success = solution is not None
                if is_success:
                    successful_solves += 1
                
                # Save individual solution
                solution_data = {
                    'filename': png_file.name,
                    'filepath': str(png_file),
                    'solution': solution or 'FAILED',
                    'solve_time': solve_time,
                    'success': is_success,
                    'confidence_threshold': args.confidence_threshold,
                    'index': i + 1,
                    'method_used': method_used or 'fallback_sequence'
                }
                
                # Save solution JSON
                solution_file = batch_solution_dir / f"{png_file.stem}_solution.json"
                with open(solution_file, 'w') as f:
                    json.dump(solution_data, f, indent=2)
                
                # Copy original image
                original_copy = originals_dir / png_file.name
                shutil.copy2(png_file, original_copy)
                
                # Save preprocessed image
                try:
                    _ = solver.preprocessor.preprocess_image(str(png_file), save_preprocessed=True, output_dir=str(preprocessed_dir))
                except Exception as e:
                    logger.debug(f"Failed to save preprocessed image for {png_file.name}: {e}")
                
                batch_results.append(solution_data)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(png_files)} files, success rate: {successful_solves/(i+1):.3f}")
            
            # Save batch results
            success_rate = successful_solves / len(png_files)
            avg_time = total_time / len(png_files)
            
            batch_summary = {
                'summary': {
                    'total_files': len(png_files),
                    'successful_solves': successful_solves,
                    'success_rate': success_rate,
                    'total_time': total_time,
                    'average_solve_time': avg_time,
                    'timestamp': timestamp
                },
                'results': batch_results
            }
            
            results_file = batch_solution_dir / "batch_solve_results.json"
            with open(results_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            # Create CSV for easy viewing
            csv_file = batch_solution_dir / "batch_solve_results.csv"
            df = pd.DataFrame(batch_results)
            df.to_csv(csv_file, index=False)
            
            print(f"\nBatch solve completed!")
            print(f"Success rate: {success_rate:.3f} ({successful_solves}/{len(png_files)})")
            print(f"Average solve time: {avg_time:.3f}s")
            print(f"Results saved to: {batch_solution_dir}")
            
            # Performance stats
            stats = solver.get_performance_stats()
            print(f"Performance stats: {stats}")
            
            # Save performance stats
            stats_file = batch_solution_dir / "performance_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        
        else:
            print(f"Invalid path: {image_path} (not a file or directory)")
            return

if __name__ == "__main__":
    main()
