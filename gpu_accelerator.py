#!/usr/bin/env python3
"""
gpu_accelerator.py - GPU Acceleration Helper for ECI CAPTCHA Solver
Provides GPU acceleration using PyTorch CUDA or TensorFlow, with fallback to CPU
"""

import logging
import warnings
from typing import Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """GPU acceleration manager for the captcha solver"""
    
    def __init__(self):
        self.device = "cpu"
        self.backend = None
        self.torch_available = False
        self.tf_gpu_available = False
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU acceleration using available backends"""
        
        # Try PyTorch CUDA first (best option for Windows)
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.backend = "pytorch"
                self.torch_available = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("GPU acceleration enabled via PyTorch CUDA")
                logger.info(f"  Device: {gpu_name}")
                logger.info(f"  CUDA version: {torch.version.cuda}")
                return
        except ImportError:
            logger.debug("PyTorch not available")
        
        # Try TensorFlow GPU as fallback
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.backend = "tensorflow"
                self.tf_gpu_available = True
                logger.info("GPU acceleration enabled via TensorFlow")
                logger.info(f"  GPUs detected: {len(gpus)}")
                return
        except ImportError:
            logger.debug("TensorFlow not available")
        
        # Fallback to CPU
        self.device = "cpu"
        self.backend = "cpu"
        logger.warning("No GPU acceleration available - using CPU")
        logger.info("For GPU acceleration, ensure CUDA drivers are installed")
    
    def get_device_info(self) -> dict:
        """Get information about the current acceleration device"""
        info = {
            "device": self.device,
            "backend": self.backend,
            "torch_available": self.torch_available,
            "tf_gpu_available": self.tf_gpu_available
        }
        
        if self.torch_available:
            import torch
            info.update({
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_version": torch.version.cuda
            })
        
        return info
    
    def to_device(self, data):
        """Move data to the appropriate device"""
        if self.backend == "pytorch" and self.torch_available:
            import torch
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.device)
            elif isinstance(data, torch.Tensor):
                return data.to(self.device)
        
        return data
    
    def to_numpy(self, data):
        """Convert data back to numpy array"""
        if self.backend == "pytorch" and self.torch_available:
            import torch
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
        
        if hasattr(data, 'numpy'):
            return data.numpy()
        
        return data
    
    def create_tensor_ops(self):
        """Create tensor operations optimized for current backend"""
        if self.backend == "pytorch":
            return PyTorchOps(self.device)
        elif self.backend == "tensorflow":
            return TensorFlowOps()
        else:
            return CPUOps()

class PyTorchOps:
    """PyTorch-based tensor operations for GPU acceleration"""
    
    def __init__(self, device="cuda"):
        import torch
        self.device = device
        self.torch = torch
    
    def resize_batch(self, images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize a batch of images using GPU acceleration"""
        import torch.nn.functional as F
        
        # Convert to PyTorch tensor
        if len(images.shape) == 3:  # Single image
            images = images[np.newaxis, ...]
        
        # Convert HWC to CHW format for PyTorch
        if images.shape[-1] in [1, 3]:  # Channels last
            images = np.transpose(images, (0, 3, 1, 2))
        
        tensor = self.torch.from_numpy(images.astype(np.float32)).to(self.device)
        
        # Resize using bilinear interpolation
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        
        # Convert back to numpy and HWC format
        result = resized.cpu().numpy()
        if result.shape[1] in [1, 3]:  # Channels first -> channels last
            result = np.transpose(result, (0, 2, 3, 1))
        
        return result.squeeze() if len(images.shape) == 3 else result
    
    def normalize_batch(self, images: np.ndarray, mean=0.5, std=0.5) -> np.ndarray:
        """Normalize a batch of images using GPU"""
        tensor = self.torch.from_numpy(images.astype(np.float32)).to(self.device)
        normalized = (tensor - mean) / std
        return normalized.cpu().numpy()
    
    def apply_gaussian_blur(self, images: np.ndarray, kernel_size=5, sigma=1.0) -> np.ndarray:
        """Apply Gaussian blur using GPU acceleration"""
        import torch.nn.functional as F
        from torch.nn import Conv2d
        import math
        
        # Create Gaussian kernel
        kernel_1d = self.torch.zeros(kernel_size).to(self.device)
        for i in range(kernel_size):
            kernel_1d[i] = math.exp(-(i - kernel_size // 2) ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)
        
        # Convert images to tensor
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        tensor = self.torch.from_numpy(images.astype(np.float32)).to(self.device)
        if len(tensor.shape) == 4 and tensor.shape[-1] in [1, 3]:
            tensor = tensor.permute(0, 3, 1, 2)
        
        # Apply convolution
        blurred = F.conv2d(tensor, kernel_2d, padding=kernel_size//2)
        
        # Convert back
        result = blurred.cpu().numpy()
        if len(result.shape) == 4 and result.shape[1] in [1, 3]:
            result = result.permute(0, 2, 3, 1).numpy()
        
        return result.squeeze() if len(images.shape) == 3 else result

class TensorFlowOps:
    """TensorFlow-based tensor operations"""
    
    def __init__(self):
        import tensorflow as tf
        self.tf = tf
    
    def resize_batch(self, images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize images using TensorFlow"""
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        resized = self.tf.image.resize(images, target_size, method='bilinear')
        return resized.numpy().squeeze() if images.shape[0] == 1 else resized.numpy()
    
    def normalize_batch(self, images: np.ndarray, mean=0.5, std=0.5) -> np.ndarray:
        """Normalize images using TensorFlow"""
        normalized = (images - mean) / std
        return normalized
    
    def apply_gaussian_blur(self, images: np.ndarray, kernel_size=5, sigma=1.0) -> np.ndarray:
        """Apply Gaussian blur using TensorFlow"""
        # Simple implementation - can be enhanced
        return images

class CPUOps:
    """CPU-based tensor operations (fallback)"""
    
    def resize_batch(self, images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize images using OpenCV (CPU)"""
        import cv2
        
        if len(images.shape) == 3:
            return cv2.resize(images, target_size)
        
        resized = []
        for img in images:
            resized.append(cv2.resize(img, target_size))
        return np.array(resized)
    
    def normalize_batch(self, images: np.ndarray, mean=0.5, std=0.5) -> np.ndarray:
        """Normalize images using NumPy"""
        return (images - mean) / std
    
    def apply_gaussian_blur(self, images: np.ndarray, kernel_size=5, sigma=1.0) -> np.ndarray:
        """Apply Gaussian blur using OpenCV"""
        import cv2
        
        if len(images.shape) == 3:
            return cv2.GaussianBlur(images, (kernel_size, kernel_size), sigma)
        
        blurred = []
        for img in images:
            blurred.append(cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma))
        return np.array(blurred)

# Global accelerator instance
_accelerator = None

def get_accelerator() -> GPUAccelerator:
    """Get the global GPU accelerator instance"""
    global _accelerator
    if _accelerator is None:
        _accelerator = GPUAccelerator()
    return _accelerator

def print_gpu_status():
    """Print current GPU acceleration status"""
    accelerator = get_accelerator()
    info = accelerator.get_device_info()
    
    print("\nGPU Acceleration Status:")
    print(f"  Backend: {info['backend']}")
    print(f"  Device: {info['device']}")
    
    if info['torch_available']:
        print(f"  PyTorch CUDA: {info['gpu_count']} GPU(s)")
        if info['gpu_name']:
            print(f"    GPU: {info['gpu_name']}")
        print(f"    CUDA: {info['cuda_version']}")
    elif info['tf_gpu_available']:
        print(f"  TensorFlow GPU support available")
    else:
        print(f"  CPU-only mode")
        print(f"    Install CUDA drivers for GPU acceleration")

if __name__ == "__main__":
    # Demo usage
    print_gpu_status()
    
    # Test basic operations
    accelerator = get_accelerator()
    ops = accelerator.create_tensor_ops()
    
    # Create test image
    test_image = np.random.rand(100, 100, 3).astype(np.float32)
    print(f"\nTesting with image shape: {test_image.shape}")
    
    # Test resize
    resized = ops.resize_batch(test_image, (80, 200))
    print(f"Resized to: {resized.shape}")
    
    # Test normalization
    normalized = ops.normalize_batch(test_image)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    print("\nGPU acceleration ready for use!")