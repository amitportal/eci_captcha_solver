#!/usr/bin/env python3
"""
scripts/batch_trainer.py - Complete Training Pipeline for ECI CAPTCHA Solver
Handles comprehensive training with GPU optimization and monitoring
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from main_solver import (
        OptimizedDataLoader, ProductionCaptchaSolver, 
        OptimizedCaptchaConfig, setup_tensorflow_gpu, get_gpu_memory_info
    )
    # Import GPU accelerator
    from gpu_accelerator import get_accelerator, print_gpu_status
    GPU_ACCELERATOR_AVAILABLE = True
except ImportError as e:
    print(f"Error importing solver modules: {e}")
    print("Make sure main_solver.py and gpu_accelerator.py are in the parent directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log successful import
logger.info("✓ GPU accelerator module imported successfully")

class BatchTrainer:
    """Complete training pipeline with monitoring and optimization"""
    
    def __init__(self, config_file: str = "config/captcha_config.json", 
                 training_config_file: str = "config/training_config.json"):
        self.base_dir = Path.cwd()
        self.config = self.load_config(config_file)
        self.training_config = self.load_training_config(training_config_file)
        self.setup_logging()
        
        # Training statistics
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "models_trained": [],
            "best_accuracy": 0.0,
            "gpu_used": False
        }
    
    def load_config(self, config_file: str) -> Dict:
        """Load main configuration"""
        config_path = self.base_dir / config_file
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Error loading config {config_path}: {e}")
        
        # Default configuration for 200x80 images
        default_config = {
            "dataset_dir": "captcha_dataset",
            "model_dir": "models",
            "csv_file": "captcha_solutions_consolidated.csv",
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.0005,
            "confidence_threshold": 0.80,
            "use_gpu_training": True,
            "use_gpu_inference": "auto",
            "img_width": 200,
            "img_height": 80
        }
        
        logger.info("Using default configuration")
        return default_config
    
    def load_training_config(self, training_config_file: str) -> Dict:
        """Load training-specific configuration"""
        config_path = self.base_dir / training_config_file
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                logger.info(f"Loaded training config from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Error loading training config {config_path}: {e}")
        
        # Default training configuration
        default_training_config = {
            "validation_split": 0.15,
            "test_split": 0.15,
            "early_stopping_patience": 15,
            "reduce_lr_patience": 7,
            "use_mixed_precision": False,
            "data_augmentation": False,
            "cache_datasets": True,
            "max_training_samples": 2000,
            "min_training_samples": 500
        }
        
        logger.info("Using default training configuration")
        return default_training_config
    
    def setup_logging(self):
        """Setup training-specific logging"""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        # Add file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Training log file: {log_file}")
    
    def validate_environment(self) -> bool:
        """Validate training environment and requirements"""
        logger.info("Validating training environment...")
        
        validation_results = {
            "dataset_exists": False,
            "csv_exists": False,
            "sufficient_data": False,
            "gpu_available": False,
            "memory_sufficient": False
        }
        
        # Check dataset directory
        dataset_dir = Path(self.config["dataset_dir"])
        if dataset_dir.exists():
            validation_results["dataset_exists"] = True
            png_files = list(dataset_dir.glob("*.png"))
            logger.info(f"Found {len(png_files)} PNG files in dataset")
            
            if len(png_files) >= self.training_config["min_training_samples"]:
                validation_results["sufficient_data"] = True
        else:
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
        
        # Check CSV file
        csv_file = Path(self.config.get("csv_file", "captcha_solutions_consolidated.csv"))
        if csv_file.exists():
            validation_results["csv_exists"] = True
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                valid_samples = len(df[df['captcha_length'] == 6])
                logger.info(f"CSV contains {valid_samples} valid samples")
                
                if valid_samples >= self.training_config["min_training_samples"]:
                    validation_results["sufficient_data"] = True
                else:
                    logger.error(f"Insufficient samples in CSV: {valid_samples} < {self.training_config['min_training_samples']}")
            except Exception as e:
                logger.error(f"Error reading CSV file: {e}")
                return False
        else:
            logger.warning(f"CSV file not found: {csv_file}")
            logger.info("Will scan dataset directory instead")
        
        # Check GPU availability using our new GPU accelerator
        if self.config.get("use_gpu_training", True):
            try:
                if GPU_ACCELERATOR_AVAILABLE:
                    accelerator = get_accelerator()
                    gpu_info = accelerator.get_device_info()
                    
                    if gpu_info["backend"] != "cpu":
                        validation_results["gpu_available"] = True
                        logger.info(f"GPU available: {gpu_info['backend']} on {gpu_info['device']}")
                        
                        # For PyTorch CUDA, check GPU memory
                        if gpu_info.get("torch_available"):
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                                    logger.info(f"GPU memory: {memory_gb:.1f} GB")
                                    validation_results["memory_sufficient"] = memory_gb > 1.0  # 1GB minimum
                                else:
                                    validation_results["memory_sufficient"] = True
                            except Exception:
                                validation_results["memory_sufficient"] = True
                        else:
                            validation_results["memory_sufficient"] = True
                    else:
                        logger.warning("No GPU detected - training will use CPU (slower)")
                else:
                    logger.warning("GPU accelerator not available - training will use CPU (slower)")
            except Exception as e:
                logger.warning(f"GPU check failed: {e}")
        
        # Summary
        logger.info("Environment validation results:")
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}: {result}")
        
        # Must have dataset and sufficient data
        return validation_results["dataset_exists"] and validation_results["sufficient_data"]
    
    def prepare_training_data(self) -> Optional[Dict]:
        """Prepare and validate training data"""
        logger.info("Preparing training data...")
        
        try:
            # Initialize data loader
            data_loader = OptimizedDataLoader(
                dataset_dir=self.config["dataset_dir"],
                csv_file=self.config.get("csv_file"),
                use_gpu=self.config.get("use_gpu_training", True)
            )
            
            # Load and prepare datasets
            datasets = data_loader.prepare_datasets(
                test_size=self.training_config.get("test_split", 0.15),
                val_size=self.training_config.get("validation_split", 0.15),
                cache_datasets=self.training_config.get("cache_datasets", True)
            )
            
            # Log dataset statistics
            stats = datasets['stats']
            logger.info(f"Dataset preparation completed:")
            logger.info(f"  Total samples: {stats['total_samples']:,}")
            logger.info(f"  Training samples: {stats['train_samples']:,}")
            logger.info(f"  Validation samples: {stats['val_samples']:,}")
            logger.info(f"  Test samples: {stats['test_samples']:,}")
            logger.info(f"  Unique solutions: {stats['unique_solutions']:,}")
            
            # Validate minimum requirements
            if stats['train_samples'] < self.training_config["min_training_samples"]:
                logger.error(f"Insufficient training samples: {stats['train_samples']} < {self.training_config['min_training_samples']}")
                return None
            
            return datasets
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return None
    
    def setup_gpu_for_training(self) -> bool:
        """Setup GPU for training with optimal configuration using new GPU accelerator"""
        if not self.config.get("use_gpu_training", True):
            logger.info("GPU training disabled in config")
            return False
        
        try:
            if not GPU_ACCELERATOR_AVAILABLE:
                logger.warning("GPU accelerator not available - using CPU")
                return False
            
            # Get GPU accelerator info
            accelerator = get_accelerator()
            gpu_info = accelerator.get_device_info()
            
            if gpu_info["backend"] == "cpu":
                logger.warning("Only CPU backend available - training will be slower")
                return False
            
            logger.info(f"✓ GPU setup successful: {gpu_info['backend']} on {gpu_info['device']}")
            
            # For PyTorch CUDA backend, we can still use TensorFlow for model training
            # PyTorch will handle preprocessing acceleration via our GPU accelerator
            if gpu_info["backend"] == "pytorch":
                logger.info("Using PyTorch CUDA for preprocessing, TensorFlow CPU for model training")
                self.training_stats["gpu_used"] = True  # For preprocessing
                return True
            
            # For TensorFlow GPU backend
            if gpu_info["backend"] == "tensorflow":
                # Enable mixed precision if configured
                if self.training_config.get("use_mixed_precision", False):
                    try:
                        import tensorflow as tf
                        from keras import mixed_precision
                        policy = mixed_precision.Policy('mixed_float16')
                        mixed_precision.set_global_policy(policy)
                        logger.info("Mixed precision training enabled")
                    except Exception as e:
                        logger.warning(f"Mixed precision setup failed: {e}")
                
                self.training_stats["gpu_used"] = True
                logger.info("TensorFlow GPU training setup completed")
                return True
            
            return False
                
        except Exception as e:
            logger.error(f"GPU setup failed: {e}")
            return False
    
    def train_models(self, datasets: Dict) -> bool:
        """Execute complete model training pipeline"""
        logger.info("Starting model training pipeline...")
        
        self.training_stats["start_time"] = datetime.now()
        
        try:
            # Initialize solver with flexible GPU usage
            # We use PyTorch CUDA for preprocessing acceleration but allow TensorFlow CPU for model training
            use_gpu_mode = "auto"  # Let solver decide based on what's available
            if not self.config.get("use_gpu_training", True):
                use_gpu_mode = "disable"
            
            solver = ProductionCaptchaSolver(
                model_dir=self.config["model_dir"],
                config_file="config/captcha_config.json",
                use_gpu=use_gpu_mode
            )
            
            # Update solver configuration
            solver.config.update({
                "epochs": self.config.get("epochs", 50),
                "batch_size": self.config.get("batch_size", 32),
                "learning_rate": self.config.get("learning_rate", 0.0005)
            })
            
            # Train models
            logger.info("Training CAPTCHA solver models...")
            solver.train_models(datasets)
            
            self.training_stats["models_trained"].append("segmentation")
            logger.info("✓ Model training completed")
            
            # Evaluate trained models
            accuracy = self.evaluate_models(solver, datasets)
            self.training_stats["best_accuracy"] = accuracy
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
        finally:
            self.training_stats["end_time"] = datetime.now()
            if self.training_stats["start_time"]:
                duration = self.training_stats["end_time"] - self.training_stats["start_time"]
                self.training_stats["total_duration"] = duration.total_seconds()
    
    def evaluate_models(self, solver: ProductionCaptchaSolver, datasets: Dict) -> float:
        """Evaluate trained models and save results"""
        logger.info("Evaluating trained models...")
        
        try:
            test_paths, test_labels = datasets['test']
            
            # Evaluate on a reasonable subset for efficiency
            max_eval_samples = min(300, len(test_paths))
            eval_paths = test_paths[:max_eval_samples]
            eval_labels = test_labels[:max_eval_samples]
            
            logger.info(f"Evaluating on {max_eval_samples} samples...")
            
            correct = 0
            total = 0
            failed_predictions = 0
            confidence_scores = []
            inference_times = []
            
            for i, (img_path, true_label) in enumerate(zip(eval_paths, eval_labels)):
                start_time = time.time()
                
                try:
                    prediction = solver.solve(
                        img_path, 
                        confidence_threshold=0.7,  # Lower threshold for evaluation
                        save_to_dataset=False  # Don't save during evaluation
                    )
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    if prediction:
                        if prediction.lower() == true_label.lower():
                            correct += 1
                        # You could extract confidence score here if available
                        confidence_scores.append(0.8)  # Placeholder
                    else:
                        failed_predictions += 1
                        confidence_scores.append(0.0)
                    
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation error on {img_path}: {e}")
                    failed_predictions += 1
                    total += 1
                    inference_times.append(0.0)
                    confidence_scores.append(0.0)
                
                # Progress update
                if (i + 1) % 50 == 0:
                    current_accuracy = correct / total if total > 0 else 0
                    logger.info(f"Evaluated {i+1}/{max_eval_samples}, accuracy: {current_accuracy:.3f}")
            
            # Calculate final metrics
            accuracy = correct / total if total > 0 else 0
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Performance stats from solver
            perf_stats = solver.get_performance_stats()
            
            # Create evaluation results
            evaluation_results = {
                "timestamp": datetime.now().isoformat(),
                "evaluation_samples": max_eval_samples,
                "accuracy": accuracy,
                "correct_predictions": correct,
                "total_predictions": total,
                "failed_predictions": failed_predictions,
                "avg_inference_time": avg_inference_time,
                "avg_confidence": avg_confidence,
                "performance_stats": perf_stats,
                "training_config": self.config,
                "training_stats": self.training_stats
            }
            
            # Save evaluation results
            results_file = Path(self.config["model_dir"]) / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation completed:")
            logger.info(f"  Accuracy: {accuracy:.3f} ({correct}/{total})")
            logger.info(f"  Failed predictions: {failed_predictions}")
            logger.info(f"  Avg inference time: {avg_inference_time:.3f}s")
            logger.info(f"  Results saved: {results_file}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0
    
    def save_training_summary(self):
        """Save comprehensive training summary"""
        try:
            summary = {
                "training_completed": datetime.now().isoformat(),
                "configuration": self.config,
                "training_configuration": self.training_config,
                "training_statistics": self.training_stats,
                "system_info": {
                    "gpu_used": self.training_stats["gpu_used"],
                    "python_version": sys.version,
                    "tensorflow_version": None
                }
            }
            
            # Add TensorFlow version
            try:
                import tensorflow as tf
                summary["system_info"]["tensorflow_version"] = tf.__version__
            except:
                pass
            
            # Save training summary
            summary_file = Path(self.config["model_dir"]) / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Training summary saved: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")
    
    def run_complete_training(self) -> bool:
        """Run complete training pipeline"""
        logger.info("="*60)
        logger.info("🚀 Starting ECI CAPTCHA Solver Training Pipeline")
        logger.info("="*60)
        
        # Print GPU status if available
        if GPU_ACCELERATOR_AVAILABLE:
            try:
                print_gpu_status()
            except Exception as e:
                logger.info(f"GPU status check failed: {e}")
        
        try:
            # Step 1: Validate environment
            if not self.validate_environment():
                logger.error("❌ Environment validation failed")
                return False
            
            logger.info("✓ Environment validation passed")
            
            # Step 2: Setup GPU
            gpu_available = self.setup_gpu_for_training()
            if gpu_available:
                logger.info("✓ GPU training setup completed")
            else:
                logger.info("⚠ CPU training mode (will be slower)")
            
            # Step 3: Prepare training data
            datasets = self.prepare_training_data()
            if not datasets:
                logger.error("❌ Data preparation failed")
                return False
            
            logger.info("✓ Training data prepared")
            
            # Step 4: Train models
            if not self.train_models(datasets):
                logger.error("❌ Model training failed")
                return False
            
            logger.info("✓ Model training completed")
            
            # Step 5: Save training summary
            self.save_training_summary()
            
            # Final success message
            duration = self.training_stats.get("total_duration", 0)
            accuracy = self.training_stats.get("best_accuracy", 0)
            
            logger.info("="*60)
            logger.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"⏱️  Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"🎯 Best accuracy: {accuracy:.1%}")
            logger.info(f"🔧 Models trained: {', '.join(self.training_stats['models_trained'])}")
            logger.info(f"💻 GPU used: {'Yes' if self.training_stats['gpu_used'] else 'No'}")
            logger.info("="*60)
            logger.info("🚀 Next steps:")
            logger.info("  1. Test the models: python main_solver.py --mode test")
            logger.info("  2. Test single image: python main_solver.py --mode solve --image-path sample.png")
            logger.info("  3. Integrate with Phase 1: Use scripts/captcha_integration.py")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="ECI CAPTCHA Solver Batch Training")
    parser.add_argument("--config", default="config/captcha_config.json", help="Main config file")
    parser.add_argument("--training-config", default="config/training_config.json", help="Training config file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment")
    parser.add_argument("--use-gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--epochs", type=int, help="Override epochs in config")
    parser.add_argument("--batch-size", type=int, help="Override batch size in config")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with minimal epochs")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = BatchTrainer(args.config, args.training_config)
        
        # Override config if needed
        if args.use_gpu:
            trainer.config["use_gpu_training"] = True
        elif args.no_gpu:
            trainer.config["use_gpu_training"] = False
        
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        
        if args.batch_size:
            trainer.config["batch_size"] = args.batch_size
        
        if args.quick_test:
            trainer.config["epochs"] = 5
            trainer.training_config["max_training_samples"] = 200
            logger.info("Quick test mode: 5 epochs, 200 samples")
        
        # Run validation only
        if args.validate_only:
            if trainer.validate_environment():
                print("✅ Environment validation passed!")
                return 0
            else:
                print("❌ Environment validation failed!")
                return 1
        
        # Run complete training
        success = trainer.run_complete_training()
        
        if success:
            print("\n🎉 Training completed successfully!")
            return 0
        else:
            print("\n❌ Training failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
