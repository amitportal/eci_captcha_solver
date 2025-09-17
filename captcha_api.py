#!/usr/bin/env python3
"""
CAPTCHA Solver API - FastAPI implementation
Exposes the CAPTCHA solving functionality via HTTP endpoints
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set TensorFlow logging levels
tf.get_logger().setLevel('ERROR')

# Additional TensorFlow logging suppression
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Enable unsafe deserialization for Keras models
keras.config.enable_unsafe_deserialization()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("captcha_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Character set definitions (same as in captcha_solver.py)
LOWERCASE = ''.join([c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in {'i', 'l', 'o'}])
DIGITS = ''.join([c for c in '0123456789' if c not in {'1', '0'}])
CHARACTERS = LOWERCASE + DIGITS
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

# Global variables for model caching and statistics
_model = None
_last_used = None
CACHE_DURATION = 600  # 10 minutes in seconds

# Statistics tracking
stats = {
    "total_requests": 0,
    "successful_solutions": 0,
    "failed_solutions": 0,
    "total_solve_time": 0.0,
    "total_model_load_time": 0.0,
    "total_image_load_time": 0.0,
    "total_processing_time": 0.0,
    "total_prediction_time": 0.0,
    "total_result_processing_time": 0.0
}

# Initialize FastAPI app
app = FastAPI(
    title="ECI CAPTCHA Solver API",
    description="API for solving ECI CAPTCHA images using deep learning models",
    version="1.0.0"
)

def get_device_info():
    """Get information about the device being used (GPU/CPU)"""
    try:
        # Try to import PyTorch for GPU acceleration
        import torch
        TORCH_AVAILABLE = torch.cuda.is_available()
        if TORCH_AVAILABLE:
            return f"PyTorch-GPU ({torch.cuda.device_count()} devices)"
    except ImportError:
        pass
    
    # Check for TensorFlow GPU
    if tf.config.list_physical_devices('GPU'):
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
        # Always load the model from the 'models/simple_crnn_200x80_raw.keras' path
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

def solve_captcha_from_array(image_array: np.ndarray) -> Dict[str, Any]:
    """Solve a CAPTCHA from a numpy array and return detailed results"""
    overall_start_time = time.time()
    
    # Update statistics
    stats["total_requests"] += 1
    
    # Load the model
    model_load_start = time.time()
    model = load_model()
    model_load_time = time.time() - model_load_start
    stats["total_model_load_time"] += model_load_time
    
    if model is None:
        logger.error("Model could not be loaded")
        stats["failed_solutions"] += 1
        stats["total_solve_time"] += time.time() - overall_start_time
        return {
            "solution": "no_solution_found",
            "success": False,
            "error": "Model could not be loaded"
        }
    
    try:
        # Process image directly without any preprocessing
        image_load_start = time.time()
        image = image_array
        image_load_time = time.time() - image_load_start
        stats["total_image_load_time"] += image_load_time
        
        if image is None:
            raise ValueError("Image array is None")
        
        # Convert to the exact format expected by the model (200x80 grayscale normalized)
        # No resizing, no processing - assume image is already correct format
        process_start = time.time()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        process_time = time.time() - process_start
        stats["total_processing_time"] += process_time
        
        # Make prediction
        prediction_start = time.time()
        prediction = model.predict(image, verbose=0)
        prediction_time = time.time() - prediction_start
        stats["total_prediction_time"] += prediction_time
        
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
        stats["total_result_processing_time"] += processing_time
        
        # Check if solution is valid (all characters recognized)
        is_valid = all(c != '?' for c in predicted_chars)
        
        # Calculate time taken
        total_time = time.time() - overall_start_time
        stats["total_solve_time"] += total_time
        
        # Update statistics
        if is_valid:
            stats["successful_solutions"] += 1
        else:
            stats["failed_solutions"] += 1
        
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
        
        # Return result
        return {
            "solution": solution if is_valid else "no_solution_found",
            "success": is_valid,
            "total_time": total_time,
            "model_load_time": model_load_time,
            "image_load_time": image_load_time,
            "processing_time": process_time,
            "prediction_time": prediction_time,
            "result_processing_time": processing_time,
            "confidence": avg_confidence,
            "device": device_info,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
            
    except Exception as e:
        total_time = time.time() - overall_start_time
        stats["total_solve_time"] += total_time
        stats["failed_solutions"] += 1
        logger.error(f"Error solving CAPTCHA: {str(e)}")
        return {
            "solution": "no_solution_found",
            "success": False,
            "error": str(e),
            "total_time": total_time,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ECI CAPTCHA Solver API",
        "version": "1.0.0",
        "description": "API for solving ECI CAPTCHA images using deep learning models",
        "endpoints": {
            "POST /solve": "Solve a CAPTCHA image",
            "GET /health": "Health check",
            "GET /stats": "Performance statistics"
        }
    }

@app.post("/solve")
async def solve_captcha(image: UploadFile = File(...)):
    """Solve a CAPTCHA image and return the solution"""
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await image.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Solve CAPTCHA
        result = solve_captcha_from_array(img)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    device_info = get_device_info()
    return {
        "status": "healthy",
        "device_info": device_info,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    # Calculate averages
    total_requests = stats["total_requests"]
    avg_solve_time = stats["total_solve_time"] / total_requests if total_requests > 0 else 0
    avg_model_load_time = stats["total_model_load_time"] / total_requests if total_requests > 0 else 0
    avg_image_load_time = stats["total_image_load_time"] / total_requests if total_requests > 0 else 0
    avg_processing_time = stats["total_processing_time"] / total_requests if total_requests > 0 else 0
    avg_prediction_time = stats["total_prediction_time"] / total_requests if total_requests > 0 else 0
    avg_result_processing_time = stats["total_result_processing_time"] / total_requests if total_requests > 0 else 0
    success_rate = stats["successful_solutions"] / total_requests if total_requests > 0 else 0
    
    return {
        "total_requests": total_requests,
        "successful_solutions": stats["successful_solutions"],
        "failed_solutions": stats["failed_solutions"],
        "average_solve_time": round(avg_solve_time, 3),
        "average_model_load_time": round(avg_model_load_time, 3),
        "average_image_load_time": round(avg_image_load_time, 3),
        "average_processing_time": round(avg_processing_time, 3),
        "average_prediction_time": round(avg_prediction_time, 3),
        "average_result_processing_time": round(avg_result_processing_time, 3),
        "success_rate": round(success_rate, 3),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

if __name__ == "__main__":
    import uvicorn
    # Run the API server
    uvicorn.run(
        "captcha_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info"
    )