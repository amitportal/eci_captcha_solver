#!/usr/bin/env python3
"""
test_framework.py - Comprehensive Testing & Validation Framework
Complete testing suite for ECI CAPTCHA Solver Phase 4 implementation
"""

import os
import sys
import json
import time
import logging
import unittest
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import GPU accelerator
try:
    from gpu_accelerator import get_accelerator, print_gpu_status
    GPU_ACCELERATOR_AVAILABLE = True
    logger.info("✓ GPU accelerator module available for testing")
except ImportError as e:
    logger.warning(f"GPU accelerator not available for testing: {e}")
    GPU_ACCELERATOR_AVAILABLE = False
    get_accelerator = None
    print_gpu_status = None

class CaptchaTestFramework:
    """Comprehensive testing framework for CAPTCHA solver"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()
        self.test_results = {}
        self.performance_metrics = {}
        
        logger.info(f"Test framework initialized in: {self.base_dir}")
    
    def run_all_tests(self, save_results: bool = True) -> Dict:
        """Run complete test suite"""
        logger.info("🧪 Starting comprehensive CAPTCHA solver testing...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_framework_version": "1.0",
            "base_directory": str(self.base_dir),
            "tests": {}
        }
        
        # Define test sequence
        test_sequence = [
            ("environment", "Environment & Dependencies", self.test_environment),
            ("dataset", "Dataset Quality & Validation", self.test_dataset_quality),
            ("preprocessing", "Image Preprocessing", self.test_preprocessing),
            ("models", "Model Functionality", self.test_model_functionality),
            ("integration", "Integration Scripts", self.test_integration),
            ("performance", "Performance Metrics", self.test_performance)
        ]
        
        # Execute tests
        for test_key, test_name, test_func in test_sequence:
            logger.info(f"Running {test_name} tests...")
            try:
                test_results["tests"][test_key] = test_func()
                status = test_results["tests"][test_key]["status"]
                status_icon = "✅" if status == "PASS" else "❌"
                logger.info(f"{status_icon} {test_name}: {status}")
            except Exception as e:
                logger.error(f"❌ {test_name} test failed: {e}")
                test_results["tests"][test_key] = {
                    "status": "ERROR",
                    "error": str(e),
                    "details": {}
                }
        
        # Overall status
        all_passed = all(
            test["status"] == "PASS" 
            for test in test_results["tests"].values()
        )
        
        test_results["overall_status"] = "PASS" if all_passed else "FAIL"
        test_results["summary"] = self._generate_test_summary(test_results["tests"])
        
        logger.info(f"🏁 Testing completed. Overall status: {test_results['overall_status']}")
        
        # Save results if requested
        if save_results:
            self._save_test_results(test_results)
        
        return test_results
    
    def test_environment(self) -> Dict:
        """Test environment setup and dependencies"""
        logger.info("Testing environment...")
        
        test_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Test Python version
        try:
            python_version = sys.version_info
            test_result["details"]["python_version"] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version < (3, 8):
                test_result["status"] = "FAIL"
                test_result["errors"].append(f"Python version {python_version} < 3.8 (minimum required)")
            elif python_version < (3, 9):
                test_result["warnings"].append("Python 3.9+ recommended for optimal performance")
                
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Python version check failed: {e}")
        
        # Test required packages
        required_packages = {
            "tensorflow": "2.12.0",
            "cv2": "4.8.0", 
            "numpy": "1.24.0",
            "pandas": "2.0.0",
            "matplotlib": "3.7.0",
            "sklearn": "1.3.0",
            "PIL": "10.0.0"
        }
        
        missing_packages = []
        package_versions = {}
        
        for package, min_version in required_packages.items():
            try:
                if package == "cv2":
                    import cv2
                    package_versions[package] = cv2.__version__
                elif package == "sklearn":
                    import sklearn
                    package_versions[package] = sklearn.__version__
                elif package == "PIL":
                    import PIL
                    package_versions[package] = PIL.__version__
                else:
                    module = __import__(package)
                    package_versions[package] = getattr(module, '__version__', 'unknown')
                
                test_result["details"][f"{package}_available"] = True
                test_result["details"][f"{package}_version"] = package_versions[package]
                
            except ImportError:
                missing_packages.append(package)
                test_result["details"][f"{package}_available"] = False
        
        if missing_packages:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Missing packages: {missing_packages}")
        
        test_result["details"]["package_versions"] = package_versions
        
        # Test GPU availability - comprehensive GPU testing
        try:
            # Test TensorFlow GPU detection
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            test_result["details"]["tensorflow_gpu_available"] = len(gpus) > 0
            test_result["details"]["tensorflow_gpu_count"] = len(gpus)
            
            if gpus:
                test_result["details"]["gpu_names"] = [gpu.name for gpu in gpus]
                logger.info(f"Found {len(gpus)} TensorFlow GPU(s)")
            else:
                test_result["warnings"].append("No TensorFlow GPU detected - will use CPU for model training")
            
            # Test GPU accelerator (primary GPU interface)
            if GPU_ACCELERATOR_AVAILABLE and get_accelerator is not None:
                try:
                    accelerator = get_accelerator()
                    gpu_info = accelerator.get_device_info()
                    test_result["details"]["gpu_accelerator_available"] = True
                    test_result["details"]["gpu_accelerator_backend"] = gpu_info["backend"]
                    test_result["details"]["gpu_accelerator_device"] = gpu_info["device"]
                    test_result["details"]["gpu_accelerator_torch_available"] = gpu_info.get("torch_available", False)
                    test_result["details"]["gpu_accelerator_tf_available"] = gpu_info.get("tf_gpu_available", False)
                    
                    if gpu_info["backend"] == "pytorch":
                        test_result["details"]["pytorch_cuda_available"] = True
                        test_result["details"]["pytorch_gpu_count"] = gpu_info.get("gpu_count", 0)
                        if gpu_info.get("gpu_name"):
                            test_result["details"]["pytorch_gpu_name"] = gpu_info["gpu_name"]
                        logger.info(f"✓ GPU accelerator: {gpu_info['backend']} on {gpu_info['device']}")
                        if gpu_info.get("gpu_name"):
                            logger.info(f"  GPU: {gpu_info['gpu_name']}")
                    elif gpu_info["backend"] == "tensorflow":
                        logger.info(f"✓ GPU accelerator: TensorFlow GPU available")
                    else:
                        test_result["warnings"].append("GPU accelerator defaulted to CPU mode")
                        
                except Exception as e:
                    test_result["details"]["gpu_accelerator_available"] = False
                    test_result["details"]["gpu_accelerator_error"] = str(e)
                    test_result["warnings"].append(f"GPU accelerator test failed: {e}")
            else:
                test_result["details"]["gpu_accelerator_available"] = False
                test_result["warnings"].append("GPU accelerator module not available - using CPU only")
                
            # Summary GPU recommendation
            if test_result["details"].get("gpu_accelerator_available") and test_result["details"]["gpu_accelerator_backend"] == "pytorch":
                test_result["details"]["gpu_recommendation"] = "✓ Optimal: PyTorch CUDA preprocessing + TensorFlow CPU training"
            elif test_result["details"].get("tensorflow_gpu_available"):
                test_result["details"]["gpu_recommendation"] = "Good: TensorFlow GPU training available"
            else:
                test_result["details"]["gpu_recommendation"] = "⚠ CPU-only mode - install CUDA drivers for acceleration"
                
        except Exception as e:
            test_result["details"]["tensorflow_gpu_available"] = False
            test_result["details"]["gpu_error"] = str(e)
            test_result["warnings"].append(f"GPU testing failed: {e}")
        
        # Test directory structure
        required_dirs = ["captcha_dataset", "models", "config", "scripts", "logs"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            exists = dir_path.exists()
            test_result["details"][f"{dir_name}_exists"] = exists
            
            if not exists:
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Missing directories: {missing_dirs}")
        
        # Test configuration files
        config_files = ["config/captcha_config.json", "config/training_config.json"]
        for config_file in config_files:
            config_path = self.base_dir / config_file
            exists = config_path.exists()
            test_result["details"][f"{config_file.replace('/', '_')}_exists"] = exists
            
            if exists:
                try:
                    with open(config_path) as f:
                        json.load(f)
                    test_result["details"][f"{config_file.replace('/', '_')}_valid"] = True
                except:
                    test_result["details"][f"{config_file.replace('/', '_')}_valid"] = False
                    test_result["warnings"].append(f"Invalid JSON in {config_file}")
        
        return test_result
    
    def test_dataset_quality(self) -> Dict:
        """Test dataset quality and consistency"""
        logger.info("Testing dataset quality...")
        
        test_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        dataset_dir = self.base_dir / "captcha_dataset"
        
        if not dataset_dir.exists():
            test_result["status"] = "FAIL"
            test_result["errors"].append("Dataset directory not found")
            return test_result
        
        # Count files
        png_files = list(dataset_dir.glob("*.png"))
        json_files = list(dataset_dir.glob("*.json"))
        
        test_result["details"]["png_count"] = len(png_files)
        test_result["details"]["json_count"] = len(json_files)
        
        logger.info(f"Found {len(png_files)} PNG files, {len(json_files)} JSON files")
        
        # Check minimum dataset size
        min_required = 100
        if len(png_files) < min_required:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Insufficient PNG files: {len(png_files)} < {min_required}")
        
        # Test file consistency and quality
        valid_pairs = 0
        invalid_solutions = 0
        corrupted_images = 0
        dimension_issues = 0
        
        # Test sample of files (max 200 for performance)
        test_files = json_files[:min(200, len(json_files))]
        
        for json_file in test_files:
            try:
                # Check JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                solution = data.get('solution', '')
                
                # Validate solution
                if len(solution) != 6 or not solution.replace(' ', '').isalnum():
                    invalid_solutions += 1
                    continue
                
                # Check corresponding PNG
                png_file = json_file.with_suffix('.png')
                if png_file.exists():
                    # Try to load and validate image
                    img = cv2.imread(str(png_file))
                    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                        height, width = img.shape[:2]
                        
                        # Check if dimensions are close to expected 200x80
                        if not (150 <= width <= 250 and 60 <= height <= 100):
                            dimension_issues += 1
                        
                        valid_pairs += 1
                    else:
                        corrupted_images += 1
                
            except Exception as e:
                logger.warning(f"Error testing {json_file}: {e}")
        
        test_result["details"]["valid_pairs"] = valid_pairs
        test_result["details"]["invalid_solutions"] = invalid_solutions
        test_result["details"]["corrupted_images"] = corrupted_images
        test_result["details"]["dimension_issues"] = dimension_issues
        test_result["details"]["tested_files"] = len(test_files)
        
        # Quality thresholds
        if valid_pairs < len(test_files) * 0.8:  # At least 80% valid
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Too few valid pairs: {valid_pairs}/{len(test_files)} < 80%")
        
        if invalid_solutions > len(test_files) * 0.1:  # Max 10% invalid solutions
            test_result["warnings"].append(f"High invalid solution rate: {invalid_solutions}/{len(test_files)}")
        
        if dimension_issues > len(test_files) * 0.2:  # Max 20% dimension issues
            test_result["warnings"].append(f"Dimension issues in {dimension_issues}/{len(test_files)} images")
        
        # Test CSV if available
        csv_file = self.base_dir / "captcha_solutions_consolidated.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                valid_csv_entries = len(df[df['captcha_length'] == 6])
                test_result["details"]["csv_valid_entries"] = valid_csv_entries
                test_result["details"]["csv_total_entries"] = len(df)
                
                if valid_csv_entries < 100:
                    test_result["status"] = "FAIL"
                    test_result["errors"].append(f"Insufficient CSV entries: {valid_csv_entries} < 100")
                    
            except Exception as e:
                test_result["warnings"].append(f"CSV validation failed: {e}")
        else:
            test_result["warnings"].append("Consolidated CSV not found - run data_consolidator.py")
        
        return test_result
    
    def test_preprocessing(self) -> Dict:
        """Test image preprocessing functionality"""
        logger.info("Testing preprocessing...")
        
        test_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Import preprocessor
            from main import AdvancedPreprocessor, OptimizedCaptchaConfig
            
            # Test both CPU and GPU preprocessing if available
            for use_gpu in [True, False]:
                try:
                    preprocessor = AdvancedPreprocessor(use_gpu=use_gpu)
                    mode = "GPU" if use_gpu else "CPU"
                    test_result["details"][f"preprocessor_init_{mode.lower()}"] = True
                    
                    # Check if GPU accelerator is being used
                    if use_gpu and hasattr(preprocessor, 'gpu_ops') and preprocessor.gpu_ops:
                        test_result["details"]["gpu_accelerated_preprocessing"] = True
                        logger.info("✓ GPU-accelerated preprocessing enabled")
                    elif use_gpu:
                        test_result["details"]["gpu_accelerated_preprocessing"] = False
                        test_result["warnings"].append("GPU requested but not available for preprocessing")
                    
                    # Test with dummy image for this mode
                    dummy_img = np.random.randint(0, 255, (80, 200, 3), dtype=np.uint8)
                    temp_file = self.base_dir / f"temp_test_image_{mode.lower()}.png"
                    
                    try:
                        cv2.imwrite(str(temp_file), dummy_img)
                        processed = preprocessor.preprocess_image(str(temp_file))
                        
                        expected_shape = (OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1)
                        test_result["details"][f"preprocessing_shape_correct_{mode.lower()}"] = processed.shape == expected_shape
                        
                        # Test normalization
                        if 0.0 <= processed.min() and processed.max() <= 1.0:
                            test_result["details"][f"normalization_correct_{mode.lower()}"] = True
                        else:
                            test_result["warnings"].append(f"{mode} normalization issue: values not in [0,1]")
                        
                        temp_file.unlink(missing_ok=True)
                        
                    except Exception as e:
                        test_result["warnings"].append(f"{mode} preprocessing test failed: {e}")
                        
                except Exception as e:
                    test_result["warnings"].append(f"{mode} preprocessor initialization failed: {e}")
            
            # Test with sample images if available
            dataset_dir = self.base_dir / "captcha_dataset"
            png_files = list(dataset_dir.glob("*.png"))
            
            if png_files:
                # Test preprocessing on sample images (use GPU if available)
                preprocessor = AdvancedPreprocessor(use_gpu=GPU_ACCELERATOR_AVAILABLE)
                test_images = png_files[:min(5, len(png_files))]
                successful_preprocessing = 0
                
                for img_path in test_images:
                    try:
                        # Test preprocessing
                        processed = preprocessor.preprocess_image(str(img_path))
                        
                        # Validate output
                        expected_shape = (OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1)
                        if processed.shape == expected_shape:
                            successful_preprocessing += 1
                        else:
                            test_result["warnings"].append(f"Unexpected output shape: {processed.shape} != {expected_shape}")
                        
                    except Exception as e:
                        logger.warning(f"Preprocessing failed for {img_path}: {e}")
                
                test_result["details"]["successful_preprocessing"] = successful_preprocessing
                test_result["details"]["tested_images"] = len(test_images)
                
                success_rate = successful_preprocessing / len(test_images)
                if success_rate < 0.8:
                    test_result["status"] = "FAIL"
                    test_result["errors"].append(f"Low preprocessing success rate: {success_rate:.1%}")
            else:
                test_result["warnings"].append("No sample images available for preprocessing test")
                    
        except ImportError as e:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Failed to import preprocessor: {e}")
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Preprocessing test failed: {e}")
        
        return test_result
    
    def test_model_functionality(self) -> Dict:
        """Test model loading and basic functionality"""
        logger.info("Testing model functionality...")
        
        test_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test solver import
            from main import ProductionCaptchaSolver, OptimizedCaptchaConfig
            
            test_result["details"]["solver_import"] = True
            
            # Test solver initialization
            solver = ProductionCaptchaSolver(str(self.base_dir / "models"))
            test_result["details"]["solver_init"] = True
            
            # Check if models are loaded
            models_loaded = (solver.crnn_model is not None or solver.segmentation_model is not None)
            test_result["details"]["models_loaded"] = models_loaded
            
            if not models_loaded:
                test_result["warnings"].append("No trained models found - run training first")
            
            # Test configuration loading
            config_valid = bool(solver.config)
            test_result["details"]["config_loaded"] = config_valid
            
            # Test with sample image if available
            dataset_dir = self.base_dir / "captcha_dataset"
            png_files = list(dataset_dir.glob("*.png"))
            
            if png_files and models_loaded:
                test_image = png_files[0]
                
                try:
                    # Test solving (may fail if models not trained)
                    solution = solver.solve(str(test_image))
                    test_result["details"]["solving_attempted"] = True
                    test_result["details"]["solution_returned"] = solution is not None
                    
                    if solution:
                        test_result["details"]["solution_length_correct"] = len(solution) == OptimizedCaptchaConfig.CAPTCHA_LENGTH
                        test_result["details"]["solution_alphanumeric"] = solution.isalnum()
                    
                except Exception as e:
                    test_result["warnings"].append(f"Solving test failed: {e}")
                    test_result["details"]["solving_error"] = str(e)
            
            # Test statistics
            try:
                stats = solver.get_performance_stats()
                test_result["details"]["statistics_available"] = bool(stats)
            except Exception as e:
                test_result["warnings"].append(f"Statistics test failed: {e}")
            
        except ImportError as e:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Import failed: {e}")
        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["errors"].append(f"Model functionality test failed: {e}")
        
        return test_result
    
    def test_integration(self) -> Dict:
        """Test integration scripts and APIs"""
        logger.info("Testing integration...")
        
        test_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Test integration script exists
        integration_script = self.base_dir / "scripts" / "captcha_integration.py"
        if integration_script.exists():
            test_result["details"]["integration_script_exists"] = True
            
            try:
                # Test import
                sys.path.append(str(self.base_dir / "scripts"))
                from old.captcha_integration import ECICaptchaIntegration, solve_captcha_image, get_solver
                
                test_result["details"]["integration_import"] = True
                
                # Test initialization
                integration = ECICaptchaIntegration(str(self.base_dir / "models"))
                test_result["details"]["integration_init"] = True
                
                # Test health check
                health = integration.health_check()
                test_result["details"]["health_check"] = health
                test_result["details"]["health_check_passed"] = all(health.values())
                
                # Test simple function
                test_result["details"]["simple_function_available"] = callable(solve_captcha_image)
                
                # Test global solver
                global_solver = get_solver()
                test_result["details"]["global_solver_available"] = global_solver is not None
                
            except Exception as e:
                test_result["status"] = "FAIL"
                test_result["errors"].append(f"Integration test failed: {e}")
        else:
            test_result["status"] = "FAIL"
            test_result["errors"].append("Integration script not found")
        
        # Test other scripts
        scripts_to_test = [
            "scripts/batch_trainer.py",
            "scripts/data_consolidator.py"
        ]
        
        for script_path in scripts_to_test:
            script = self.base_dir / script_path
            script_name = script.name.replace('.py', '')
            test_result["details"][f"{script_name}_exists"] = script.exists()
            
            if script.exists():
                # Test if script is executable/importable
                try:
                    with open(script) as f:
                        content = f.read()
                    
                    # Basic validation
                    has_main = "def main(" in content
                    has_argparse = "argparse" in content
                    
                    test_result["details"][f"{script_name}_has_main"] = has_main
                    test_result["details"][f"{script_name}_has_argparse"] = has_argparse
                    
                except Exception as e:
                    test_result["warnings"].append(f"Error reading {script_path}: {e}")
        
        return test_result
    
    def test_performance(self) -> Dict:
        """Test performance metrics"""
        logger.info("Testing performance...")
        
        test_result = {
            "status": "PASS",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            test_result["details"]["memory_usage_mb"] = memory_mb
            
            # Test with small sample if dataset available
            dataset_dir = self.base_dir / "captcha_dataset"
            png_files = list(dataset_dir.glob("*.png"))
            
            if len(png_files) >= 5:
                from main import ProductionCaptchaSolver
                
                solver = ProductionCaptchaSolver(str(self.base_dir / "models"))
                
                # Time inference attempts
                times = []
                successes = 0
                test_files = png_files[:5]  # Test 5 files
                
                for img_path in test_files:
                    start_time = time.time()
                    try:
                        result = solver.solve(str(img_path))
                        end_time = time.time()
                        
                        inference_time = end_time - start_time
                        times.append(inference_time)
                        
                        if result:
                            successes += 1
                            
                    except Exception as e:
                        logger.warning(f"Performance test error on {img_path}: {e}")
                        times.append(float('inf'))
                
                if times:
                    # Filter out infinite times
                    valid_times = [t for t in times if t != float('inf')]
                    
                    if valid_times:
                        avg_time = np.mean(valid_times)
                        max_time = np.max(valid_times)
                        min_time = np.min(valid_times)
                        
                        test_result["details"]["avg_inference_time"] = avg_time
                        test_result["details"]["max_inference_time"] = max_time
                        test_result["details"]["min_inference_time"] = min_time
                        test_result["details"]["success_rate"] = successes / len(test_files)
                        test_result["details"]["tested_files"] = len(test_files)
                        
                        # Performance thresholds
                        if avg_time > 10.0:  # 10 seconds max average
                            test_result["warnings"].append(f"Slow inference: {avg_time:.2f}s > 10.0s")
                        
                        if successes == 0:
                            test_result["warnings"].append("No successful predictions (models may not be trained)")
                    else:
                        test_result["warnings"].append("All inference attempts failed")
                
                # Test GPU memory if available
                try:
                    from main import get_gpu_memory_info
                    gpu_memory = get_gpu_memory_info()
                    if gpu_memory and gpu_memory.get('total', 0) > 0:
                        test_result["details"]["gpu_memory"] = gpu_memory
                        logger.info(f"GPU memory: {gpu_memory['total']} MB total, {gpu_memory['free']} MB free")
                except:
                    pass
                
            else:
                test_result["warnings"].append("Insufficient test data for performance testing")
                
        except ImportError:
            test_result["warnings"].append("psutil not available for memory testing")
        except Exception as e:
            test_result["warnings"].append(f"Performance test failed: {e}")
        
        return test_result
    
    def _generate_test_summary(self, tests: Dict) -> Dict:
        """Generate test summary statistics"""
        summary = {
            "total_tests": len(tests),
            "passed_tests": 0,
            "failed_tests": 0,
            "error_tests": 0,
            "total_errors": 0,
            "total_warnings": 0
        }
        
        for test_name, test_result in tests.items():
            status = test_result.get("status", "UNKNOWN")
            
            if status == "PASS":
                summary["passed_tests"] += 1
            elif status == "FAIL":
                summary["failed_tests"] += 1
            elif status == "ERROR":
                summary["error_tests"] += 1
            
            summary["total_errors"] += len(test_result.get("errors", []))
            summary["total_warnings"] += len(test_result.get("warnings", []))
        
        return summary
    
    def _save_test_results(self, test_results: Dict):
        """Save test results to files"""
        try:
            # Create output directory
            output_dir = self.base_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Save JSON results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"test_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Test results saved: {results_file}")
            
            # Create markdown report
            report_content = self._create_markdown_report(test_results)
            report_file = output_dir / f"test_report_{timestamp}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Test report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
    
    def _create_markdown_report(self, test_results: Dict) -> str:
        """Create detailed markdown test report"""
        
        lines = [
            "# ECI CAPTCHA Solver - Test Report",
            f"**Generated**: {test_results['timestamp']}",
            f"**Overall Status**: {test_results['overall_status']}",
            "",
            "## Summary",
            ""
        ]
        
        summary = test_results["summary"]
        lines.extend([
            f"- **Total Tests**: {summary['total_tests']}",
            f"- **Passed**: {summary['passed_tests']} ✅",
            f"- **Failed**: {summary['failed_tests']} ❌", 
            f"- **Errors**: {summary['error_tests']} ⚠️",
            f"- **Total Warnings**: {summary['total_warnings']}",
            f"- **Total Errors**: {summary['total_errors']}",
            "",
            "## Test Results",
            ""
        ])
        
        # Individual test results
        for test_name, result in test_results["tests"].items():
            status = result["status"]
            status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"}.get(status, "❓")
            
            lines.extend([
                f"### {test_name.title()} Test {status_emoji}",
                f"**Status**: {status}",
                ""
            ])
            
            # Details
            if result.get("details"):
                lines.append("**Details**:")
                for key, value in result["details"].items():
                    lines.append(f"- `{key}`: {value}")
                lines.append("")
            
            # Errors
            if result.get("errors"):
                lines.append("**Errors**:")
                for error in result["errors"]:
                    lines.append(f"- ❌ {error}")
                lines.append("")
            
            # Warnings
            if result.get("warnings"):
                lines.append("**Warnings**:")
                for warning in result["warnings"]:
                    lines.append(f"- ⚠️ {warning}")
                lines.append("")
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            ""
        ])
        
        failed_tests = [name for name, result in test_results["tests"].items() if result["status"] == "FAIL"]
        
        if not failed_tests:
            lines.append("🎉 All tests passed! The CAPTCHA solver is ready for production use.")
        else:
            lines.extend([
                "❌ Some tests failed. Please address the following issues:",
                ""
            ])
            
            for test_name in failed_tests:
                result = test_results["tests"][test_name]
                lines.append(f"**{test_name.title()}**:")
                for error in result.get("errors", []):
                    lines.append(f"- Fix: {error}")
                lines.append("")
        
        # Next steps
        lines.extend([
            "## Next Steps",
            "",
            "1. **If tests passed**: Continue with training or integration",
            "2. **If tests failed**: Address errors and re-run tests",
            "3. **Training**: Run `python scripts/batch_trainer.py`",
            "4. **Integration**: Use `scripts/captcha_integration.py`",
            "5. **Performance**: Monitor with regular testing",
            ""
        ])
        
        return "\n".join(lines)

class CaptchaUnitTests(unittest.TestCase):
    """Unit tests for individual components"""
    
    def setUp(self):
        self.base_dir = Path(".")
        
    def test_config_loading(self):
        """Test configuration file loading"""
        config_file = self.base_dir / "config" / "captcha_config.json"
        
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            
            # Test required fields
            self.assertIn("confidence_threshold", config)
            self.assertIsInstance(config["confidence_threshold"], (int, float))
            self.assertTrue(0.0 <= config["confidence_threshold"] <= 1.0)
            
            if "img_width" in config:
                self.assertEqual(config["img_width"], 200)
            
            if "img_height" in config:
                self.assertEqual(config["img_height"], 80)
    
    def test_character_encoding(self):
        """Test character encoding/decoding"""
        try:
            sys.path.append(str(self.base_dir))
            from main import OptimizedCaptchaConfig
            
            # Test all characters can be encoded and decoded
            for i, char in enumerate(OptimizedCaptchaConfig.CHARACTERS):
                encoded = OptimizedCaptchaConfig.CHAR_TO_NUM[char]
                decoded = OptimizedCaptchaConfig.NUM_TO_CHAR[encoded]
                
                self.assertEqual(char, decoded)
                self.assertEqual(i, encoded)
            
            # Test total character count
            self.assertEqual(len(OptimizedCaptchaConfig.CHARACTERS), 62)
            
        except ImportError:
            self.skipTest("Solver module not available")
    
    def test_image_preprocessing(self):
        """Test image preprocessing with dummy data"""
        try:
            sys.path.append(str(self.base_dir))
            from main import AdvancedPreprocessor, OptimizedCaptchaConfig
            
            preprocessor = AdvancedPreprocessor()
            
            # Create dummy 200x80 image
            dummy_image = np.random.randint(0, 255, (80, 200, 3), dtype=np.uint8)
            temp_file = "temp_test_image.png"
            
            try:
                cv2.imwrite(temp_file, dummy_image)
                processed = preprocessor.preprocess_image(temp_file)
                
                # Test output shape
                expected_shape = (OptimizedCaptchaConfig.IMG_HEIGHT, OptimizedCaptchaConfig.IMG_WIDTH, 1)
                self.assertEqual(processed.shape, expected_shape)
                
                # Test value range [0, 1]
                self.assertGreaterEqual(processed.min(), 0.0)
                self.assertLessEqual(processed.max(), 1.0)
                
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except ImportError:
            self.skipTest("Solver module not available")

def create_sample_dataset(output_dir: str, num_samples: int = 20):
    """Create sample dataset for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Creating sample dataset with {num_samples} samples...")
    
    import random
    import string
    
    for i in range(num_samples):
        # Generate random 6-character solution
        solution = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        
        # Create dummy 200x80 image
        dummy_img = np.random.randint(0, 255, (80, 200, 3), dtype=np.uint8)
        
        # Add some text-like patterns (simple simulation)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(dummy_img, solution.upper(), (10, 50), font, 1.5, (0, 255, 0), 2)
        
        # Add some noise lines (simulation)
        cv2.line(dummy_img, (0, 30), (200, 30), (255, 255, 0), 2)  # Yellow line
        cv2.line(dummy_img, (0, 60), (200, 60), (0, 255, 255), 2)  # Cyan line
        cv2.line(dummy_img, (20, 0), (180, 80), (0, 255, 0), 1)    # Diagonal green line
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{solution.upper()}_{timestamp}_TEST_{i:03d}"
        
        # Save image
        img_file = output_path / f"{base_name}.png"
        cv2.imwrite(str(img_file), dummy_img)
        
        # Save JSON metadata
        json_file = output_path / f"{base_name}.json"
        metadata = {
            "solution": solution,
            "timestamp": datetime.now().isoformat(),
            "source": "test_generator",
            "test_sample": True,
            "image_dimensions": "200x80"
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Sample dataset created in {output_path}")
    return output_path

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="ECI CAPTCHA Solver Testing Framework")
    parser.add_argument("--base-dir", default=".", help="Base directory")
    parser.add_argument("--test-type", choices=["all", "unit", "integration", "performance", "environment", "dataset"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--create-sample-dataset", type=int, metavar="N", 
                       help="Create sample dataset with N samples")
    parser.add_argument("--save-results", action="store_true", help="Save test results to files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.create_sample_dataset:
        dataset_dir = Path(args.base_dir) / "captcha_dataset"
        create_sample_dataset(str(dataset_dir), args.create_sample_dataset)
        print(f"✅ Sample dataset created with {args.create_sample_dataset} samples")
        return 0
    
    if args.test_type == "unit":
        # Run unit tests
        print("🧪 Running unit tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(CaptchaUnitTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ All unit tests passed!")
            return 0
        else:
            print(f"❌ Unit tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            return 1
    
    else:
        # Run comprehensive tests
        framework = CaptchaTestFramework(args.base_dir)
        
        if args.test_type == "all":
            test_results = framework.run_all_tests(save_results=args.save_results)
        else:
            # Run specific test
            test_func = getattr(framework, f"test_{args.test_type}", None)
            if test_func:
                print(f"🧪 Running {args.test_type} tests...")
                result = test_func()
                
                test_results = {
                    "timestamp": datetime.now().isoformat(),
                    "tests": {args.test_type: result},
                    "overall_status": result["status"]
                }
            else:
                print(f"❌ Unknown test type: {args.test_type}")
                return 1
        
        # Print results summary
        print(f"\n📊 Test Results Summary:")
        print(f"Overall Status: {test_results['overall_status']}")
        
        for test_name, result in test_results["tests"].items():
            status_symbol = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_symbol} {test_name.title()}: {result['status']}")
            
            # Show errors
            if result.get("errors"):
                for error in result["errors"]:
                    print(f"   ❌ {error}")
            
            # Show warnings  
            if result.get("warnings"):
                for warning in result["warnings"]:
                    print(f"   ⚠️ {warning}")
        
        # Exit with appropriate code
        if test_results["overall_status"] == "PASS":
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

if __name__ == "__main__":
    exit(main())
