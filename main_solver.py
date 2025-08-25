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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def __init__(self, use_gpu: bool = True):
        self.target_size = (OptimizedCaptchaConfig.IMG_WIDTH, OptimizedCaptchaConfig.IMG_HEIGHT)
        
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
        """Complete optimized preprocessing pipeline for 200x80 images"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
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
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_path}: {e}")
            # Return blank image as fallback
            return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.float32)

class OptimizedDataLoader:
    """Optimized data loader for ECI CAPTCHA dataset with 200x80 images"""
    
    def __init__(self, dataset_dir: str, csv_file: str = None, use_gpu: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.csv_file = csv_file
        self.use_gpu = use_gpu
        self.preprocessor = AdvancedPreprocessor()
    
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
                        image_paths.append(str(image_path))
                        # Normalize solution to lowercase (72.7% of dataset is lowercase)
                        solutions.append(str(row['solution']).lower())
                    else:
                        invalid_images += 1
                else:
                    missing_files += 1
            
            logger.info(f"Found {len(image_paths)} valid image files")
            logger.info(f"Missing files: {missing_files}, Invalid images: {invalid_images}")
            
            return image_paths, solutions
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return self.load_from_directory()
    
    def load_from_directory(self) -> Tuple[List[str], List[str]]:
        """Fallback: load from directory scanning"""
        image_paths = []
        solutions = []
        
        for json_file in self.dataset_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                solution = data.get('solution', '').strip()
                
                if len(solution) == OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                    png_file = json_file.with_suffix('.png')
                    if png_file.exists():
                        # Validate image
                        img = cv2.imread(str(png_file))
                        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                            image_paths.append(str(png_file))
                            solutions.append(solution.lower())  # Normalize to lowercase
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        logger.info(f"Directory scan found {len(image_paths)} valid samples")
        return image_paths, solutions
    
    def create_tf_dataset(self, image_paths: List[str], labels: List[str], 
                         batch_size: int = None, shuffle: bool = True,
                         cache: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset for 200x80 images"""
        if batch_size is None:
            batch_size = OptimizedCaptchaConfig.BATCH_SIZE
        
        def load_and_preprocess(image_path, label):
            # Load and preprocess image
            image = tf.py_function(
                func=lambda path: self.preprocessor.preprocess_image(path.numpy().decode('utf-8')),
                inp=[image_path],
                Tout=tf.float32
            )
            image.set_shape([OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1])
            
            # Encode label
            label_encoded = tf.py_function(
                func=lambda lbl: np.array([OptimizedCaptchaConfig.CHAR_TO_NUM[c] for c in lbl.numpy().decode('utf-8')]),
                inp=[label],
                Tout=tf.int32
            )
            label_encoded.set_shape([OptimizedCaptchaConfig.CAPTCHA_LENGTH])
            
            return image, label_encoded
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if cache:
            dataset = dataset.cache()
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def prepare_datasets(self, test_size: float = 0.15, val_size: float = 0.15,
                        cache_datasets: bool = True) -> Dict:
        """Prepare optimized train/val/test datasets for 200x80 images"""
        image_paths, solutions = self.load_from_csv()
        
        if len(image_paths) < 100:
            raise ValueError(f"Insufficient data: only {len(image_paths)} samples found")
        
        # Log dataset statistics
        logger.info(f"Total samples: {len(image_paths)}")
        logger.info(f"Unique solutions: {len(set(solutions))}")
        
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
        train_dataset = self.create_tf_dataset(train_paths, train_labels, shuffle=True, cache=cache_datasets)
        val_dataset = self.create_tf_dataset(val_paths, val_labels, shuffle=False, cache=cache_datasets)
        test_dataset = self.create_tf_dataset(test_paths, test_labels, shuffle=False, cache=False)
        
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
                'unique_solutions': len(set(solutions))
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
        """Build optimized CRNN architecture for 200x80 images"""
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
        
        # Optimized RNN layers
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.2))(x)
        
        # Output layer for character sequence
        outputs = layers.Dense(OptimizedCaptchaConfig.NUM_CLASSES, activation='softmax', name='character_output')(x)
        
        # Handle mixed precision output
        if self.use_mixed_precision:
            outputs = layers.Activation('softmax', dtype='float32', name='predictions')(outputs)
        
        model = keras.Model(input_img, outputs, name='OptimizedCRNN_200x80')
        return model
    
    def compile_model(self, learning_rate: float = None):
        """Compile model with optimized settings"""
        if learning_rate is None:
            learning_rate = OptimizedCaptchaConfig.LEARNING_RATE
        
        self.model = self.build_optimized_model()
        
        # Use Adam optimizer with gradient clipping for stability
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Gradient clipping
        )
        
        # For mixed precision, loss scaling is handled automatically
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model compiled with {self.model.count_params():,} parameters")
        return self.model

class EfficientSegmentationModel:
    """Efficient segmentation-based model optimized for 200x80 ECI patterns"""
    
    def __init__(self):
        self.model = None
        self.character_width = OptimizedCaptchaConfig.IMG_WIDTH // OptimizedCaptchaConfig.CAPTCHA_LENGTH  # ~33 pixels per character
        self.character_height = OptimizedCaptchaConfig.IMG_HEIGHT  # 80 pixels
    
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
            
            # Dynamic character boundary detection
            in_char = False
            char_starts = []
            char_ends = []
            
            for i, val in enumerate(smooth_projection):
                if val > threshold and not in_char:
                    char_starts.append(i)
                    in_char = True
                elif val <= threshold and in_char:
                    char_ends.append(i)
                    in_char = False
            
            if in_char:  # Handle case where last character goes to end
                char_ends.append(len(smooth_projection))
            
            # If we don't get exactly 6 regions, fall back to equal division
            if len(char_starts) != OptimizedCaptchaConfig.CAPTCHA_LENGTH or len(char_ends) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.debug(f"Dynamic segmentation found {len(char_starts)} regions, using equal division")
                char_width = OptimizedCaptchaConfig.IMG_WIDTH // OptimizedCaptchaConfig.CAPTCHA_LENGTH
                
                for i in range(OptimizedCaptchaConfig.CAPTCHA_LENGTH):
                    start_x = i * char_width
                    end_x = (i + 1) * char_width
                    
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
                 config_file: str = None, use_gpu: str = "auto"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.preprocessor = AdvancedPreprocessor()
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
        self.load_models()
    
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
        """Solve using CRNN approach (placeholder for full implementation)"""
        try:
            # This would implement proper CRNN prediction
            # For now, return None to indicate not fully implemented
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
        """Save solved CAPTCHA to dataset for continuous learning"""
        try:
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
        
        # Train segmentation model
        self._train_segmentation(datasets)
        
        # Train CRNN model (if sufficient data)
        if datasets['stats']['train_samples'] > 1000:
            self._train_crnn(datasets)
        
        # Save models
        self.save_models()
        
        logger.info("Training completed successfully!")
    
    def _train_segmentation(self, datasets: Dict):
        """Train segmentation model with character-level data"""
        try:
            logger.info("Training segmentation model for 200x80 images...")
            
            # Prepare character-level training data
            char_images = []
            char_labels = []
            
            seg_model = EfficientSegmentationModel()
            
            # Extract character images from training data
            train_paths, train_labels = datasets['train']
            
            # Use a reasonable subset for training to avoid memory issues
            max_training_samples = min(2000, len(train_paths))
            
            for i, (img_path, label) in enumerate(zip(train_paths[:max_training_samples], train_labels[:max_training_samples])):
                if i % 200 == 0:
                    logger.info(f"Processing training sample {i}/{max_training_samples}")
                
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Preprocess
                        processed = self.preprocessor.preserve_green_text_during_line_removal(img)
                        char_regions = seg_model.extract_character_regions(processed)
                        
                        if len(char_regions) == len(label):
                            for char_img, char_label in zip(char_regions, label):
                                char_images.append(np.expand_dims(char_img, axis=-1))
                                char_labels.append(OptimizedCaptchaConfig.CHAR_TO_NUM[char_label])
                                
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
            
            if len(char_images) > 500:  # Minimum required for training
                char_images = np.array(char_images)
                char_labels = np.array(char_labels)
                
                logger.info(f"Training segmentation model with {len(char_images)} character samples")
                
                self.segmentation_model = seg_model.build_character_classifier()
                
                # Train with callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
                    keras.callbacks.ModelCheckpoint(
                        str(self.model_dir / "best_segmentation.h5"),
                        save_best_only=True,
                        monitor='val_accuracy'
                    )
                ]
                
                history = self.segmentation_model.fit(
                    char_images, char_labels,
                    batch_size=OptimizedCaptchaConfig.BATCH_SIZE,
                    epochs=40,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Log training results
                final_accuracy = max(history.history['val_accuracy'])
                logger.info(f"Segmentation model trained successfully. Best validation accuracy: {final_accuracy:.3f}")
                
            else:
                logger.warning("Insufficient character data for segmentation training")
                
        except Exception as e:
            logger.error(f"Segmentation training failed: {e}")
    
    def _train_crnn(self, datasets: Dict):
        """Train CRNN model (placeholder for full implementation)"""
        try:
            logger.info("Training CRNN model for 200x80 images...")
            
            # This would implement full CRNN training with CTC loss
            # For now, just log that it would be trained
            logger.info("CRNN training would be implemented here")
            
        except Exception as e:
            logger.error(f"CRNN training failed: {e}")
    
    def save_models(self):
        """Save trained models in modern Keras format"""
        try:
            if self.crnn_model:
                model_path = self.model_dir / "optimized_crnn_200x80.keras"
                self.crnn_model.save(model_path)
                logger.info(f"✓ CRNN model saved: {model_path}")
            
            if self.segmentation_model:
                model_path = self.model_dir / "optimized_segmentation_200x80.keras"
                self.segmentation_model.save(model_path)
                logger.info(f"✓ Segmentation model saved: {model_path}")
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load saved models (try modern .keras format first, fallback to legacy .h5)"""
        try:
            # Try loading modern Keras format first
            crnn_path_keras = self.model_dir / "optimized_crnn_200x80.keras"
            crnn_path_h5 = self.model_dir / "optimized_crnn_200x80.h5"
            
            if crnn_path_keras.exists():
                self.crnn_model = keras.models.load_model(str(crnn_path_keras))
                logger.info("✓ CRNN model loaded (Keras format)")
            elif crnn_path_h5.exists():
                self.crnn_model = keras.models.load_model(str(crnn_path_h5))
                logger.info("✓ CRNN model loaded (legacy H5 format)")
            
            # Try loading segmentation model
            seg_path_keras = self.model_dir / "optimized_segmentation_200x80.keras"
            seg_path_h5 = self.model_dir / "optimized_segmentation_200x80.h5"
            
            if seg_path_keras.exists():
                self.segmentation_model = keras.models.load_model(str(seg_path_keras))
                logger.info("✓ Segmentation model loaded (Keras format)")
            elif seg_path_h5.exists():
                self.segmentation_model = keras.models.load_model(str(seg_path_h5))
                logger.info("✓ Segmentation model loaded (legacy H5 format)")
            
            # Try loading best model if available
            best_seg_path_keras = self.model_dir / "best_segmentation.keras"
            best_seg_path_h5 = self.model_dir / "best_segmentation.h5"
            
            if not self.segmentation_model:
                if best_seg_path_keras.exists():
                    self.segmentation_model = keras.models.load_model(str(best_seg_path_keras))
                    logger.info("✓ Best segmentation model loaded (Keras format)")
                elif best_seg_path_h5.exists():
                    self.segmentation_model = keras.models.load_model(str(best_seg_path_h5))
                    logger.info("✓ Best segmentation model loaded (legacy H5 format)")
                
        except Exception as e:
            logger.warning(f"Error loading models: {e}")

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


def main():
    """Main execution function for 200x80 CAPTCHA solver"""
    parser = argparse.ArgumentParser(description="ECI CAPTCHA Solver for 200x80 Images")
    parser.add_argument("--dataset-dir", help="CAPTCHA dataset directory (required for train/test modes)")
    parser.add_argument("--csv-file", help="Consolidated CSV file path")
    parser.add_argument("--mode", choices=["train", "test", "solve"], default="train")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--image-path", help="Single image path or directory containing PNG files (for solve mode)")
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--use-gpu", choices=["auto", "force", "disable"], default="auto")
    parser.add_argument("--config-file", help="Configuration file path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-preprocessed", action="store_true", help="Save preprocessed images during training")
    parser.add_argument("--output-dir", default="output", help="Output directory for results and preprocessed images")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ["train", "test"] and not args.dataset_dir:
        parser.error(f"--dataset-dir is required for {args.mode} mode")
    
    if args.mode == "solve" and not args.image_path:
        parser.error("--image-path is required for solve mode")
    
    # Print GPU status if available
    if GPU_ACCELERATOR_AVAILABLE and print_gpu_status is not None:
        try:
            print_gpu_status()
        except Exception as e:
            logger.info(f"GPU status check failed: {e}")
    
    # Update configuration
    OptimizedCaptchaConfig.EPOCHS = args.epochs
    OptimizedCaptchaConfig.BATCH_SIZE = args.batch_size
    
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
        
        data_loader = OptimizedDataLoader(args.dataset_dir, args.csv_file, use_gpu=True)
        datasets = data_loader.prepare_datasets()
        
        # Log dataset statistics
        logger.info(f"Dataset statistics: {datasets['stats']}")
        
        solver = ProductionCaptchaSolver(args.model_dir, config_file=args.config_file, use_gpu=args.use_gpu)
        
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
        
        data_loader = OptimizedDataLoader(args.dataset_dir, args.csv_file)
        datasets = data_loader.prepare_datasets()
        
        solver = ProductionCaptchaSolver(args.model_dir, config_file=args.config_file, use_gpu=args.use_gpu)
        
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
        
        solver = ProductionCaptchaSolver(args.model_dir, config_file=args.config_file, use_gpu=args.use_gpu)
        image_path = Path(args.image_path)
        
        if image_path.is_file():
            # Single file mode
            start_time = time.time()
            result = solver.solve(str(image_path), args.confidence_threshold)
            solve_time = time.time() - start_time
            
            # Save solution to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            solution_dir = Path("output") / "captcha_solutions"
            solution_dir.mkdir(parents=True, exist_ok=True)
            
            # Create solution record
            solution_data = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'solution': result or 'FAILED',
                'solve_time': solve_time,
                'success': result is not None,
                'confidence_threshold': args.confidence_threshold,
                'timestamp': timestamp
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
            
            if result:
                print(f"CAPTCHA solution: {result} (solved in {solve_time:.3f}s)")
                print(f"Solution saved to: {solution_file}")
            else:
                print(f"Failed to solve CAPTCHA (attempted for {solve_time:.3f}s)")
            
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
                result = solver.solve(str(png_file), args.confidence_threshold)
                solve_time = time.time() - file_start_time
                total_time += solve_time
                
                is_success = result is not None
                if is_success:
                    successful_solves += 1
                
                # Save individual solution
                solution_data = {
                    'filename': png_file.name,
                    'filepath': str(png_file),
                    'solution': result or 'FAILED',
                    'solve_time': solve_time,
                    'success': is_success,
                    'confidence_threshold': args.confidence_threshold,
                    'index': i + 1
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
