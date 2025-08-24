#!/usr/bin/env python3
"""
scripts/captcha_integration.py - Integration with Phase 1 Downloader
Drop-in replacement for AntiCaptcha API with ML-based solving
"""

import sys
import os
import json
import logging
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from main_solver import ProductionCaptchaSolver, OptimizedCaptchaConfig
except ImportError as e:
    print(f"Error importing solver modules: {e}")
    print("Make sure main_solver.py is in the parent directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CAPTCHA - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ECICaptchaIntegration:
    """Integration wrapper for ECI CAPTCHA solver with Phase 1 downloader"""
    
    def __init__(self, model_dir: str = "models", config_file: str = None, 
                 fallback_api_key: str = None, save_dataset: bool = True):
        """
        Initialize CAPTCHA integration
        
        Args:
            model_dir: Directory containing trained models
            config_file: Configuration file path
            fallback_api_key: AntiCaptcha API key for fallback
            save_dataset: Whether to save solved CAPTCHAs for continuous learning
        """
        self.model_dir = Path(model_dir)
        self.config_file = config_file
        self.fallback_api_key = fallback_api_key
        self.save_dataset = save_dataset
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize solver
        self.solver = None
        self._initialize_solver()
        
        # Performance tracking
        self.session_stats = {
            "session_start": datetime.now().isoformat(),
            "total_attempts": 0,
            "ml_successes": 0,
            "api_fallbacks": 0,
            "failures": 0,
            "avg_solve_time": 0.0,
            "saved_to_dataset": 0
        }
        
        logger.info(f"CAPTCHA integration initialized (models: {self.model_dir})")
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "confidence_threshold": 0.80,
            "max_retries": 3,
            "api_timeout": 30,
            "save_solved_captchas": True,
            "dataset_dir": "captcha_dataset",
            "use_gpu_inference": "auto"
        }
        
        if self.config_file:
            config_path = Path(self.config_file)
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        file_config = json.load(f)
                    default_config.update(file_config)
                    logger.info(f"Loaded config from {config_path}")
                except Exception as e:
                    logger.warning(f"Error loading config file: {e}, using defaults")
        
        return default_config
    
    def _initialize_solver(self):
        """Initialize the CAPTCHA solver"""
        try:
            self.solver = ProductionCaptchaSolver(
                model_dir=str(self.model_dir),
                fallback_api_key=self.fallback_api_key,
                config_file=self.config_file,
                use_gpu=self.config.get("use_gpu_inference", "auto")
            )
            
            # Check if models are loaded
            if self.solver.segmentation_model or self.solver.crnn_model:
                logger.info("✓ ML models loaded successfully")
            else:
                logger.warning("⚠ No trained models found - will use fallback methods")
            
        except Exception as e:
            logger.error(f"Failed to initialize solver: {e}")
            self.solver = None
    
    def solve_captcha(self, image_path: str, confidence_threshold: float = None,
                     max_retries: int = None, save_to_dataset: bool = None) -> Optional[str]:
        """
        Solve CAPTCHA image with comprehensive fallback strategy
        
        Args:
            image_path: Path to CAPTCHA image file
            confidence_threshold: Minimum confidence for ML prediction
            max_retries: Maximum retry attempts
            save_to_dataset: Whether to save solved CAPTCHA for learning
            
        Returns:
            CAPTCHA solution string or None if failed
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.get("confidence_threshold", 0.80)
        
        if max_retries is None:
            max_retries = self.config.get("max_retries", 3)
        
        if save_to_dataset is None:
            save_to_dataset = self.save_dataset and self.config.get("save_solved_captchas", True)
        
        start_time = time.time()
        self.session_stats["total_attempts"] += 1
        
        # Validate image file
        if not Path(image_path).exists():
            logger.error(f"CAPTCHA image not found: {image_path}")
            self._update_session_stats("failure", time.time() - start_time)
            return None
        
        logger.info(f"Solving CAPTCHA: {Path(image_path).name}")
        
        # Attempt solving with retries
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                
                solution = self._attempt_solve(image_path, confidence_threshold)
                
                if solution:
                    solve_time = time.time() - start_time
                    logger.info(f"✓ CAPTCHA solved: '{solution}' (time: {solve_time:.2f}s)")
                    
                    # Save to dataset for continuous learning
                    if save_to_dataset:
                        self._save_to_dataset(image_path, solution)
                    
                    self._update_session_stats("success", solve_time)
                    return solution
                else:
                    logger.warning(f"Solve attempt {attempt + 1} failed")
                    
            except Exception as e:
                logger.error(f"Error in solve attempt {attempt + 1}: {e}")
        
        # All attempts failed
        total_time = time.time() - start_time
        logger.error(f"❌ Failed to solve CAPTCHA after {max_retries} attempts (time: {total_time:.2f}s)")
        self._update_session_stats("failure", total_time)
        return None
    
    def _attempt_solve(self, image_path: str, confidence_threshold: float) -> Optional[str]:
        """Single attempt to solve CAPTCHA"""
        if not self.solver:
            logger.warning("Solver not initialized, attempting to reinitialize...")
            self._initialize_solver()
            if not self.solver:
                return None
        
        try:
            # Use the main solver
            solution = self.solver.solve(
                image_path=image_path,
                confidence_threshold=confidence_threshold,
                save_to_dataset=False  # We handle dataset saving here
            )
            
            if solution and len(solution) == OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                return solution.lower()  # Normalize to lowercase
            
            return None
            
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return None
    
    def _save_to_dataset(self, image_path: str, solution: str):
        """Save solved CAPTCHA to dataset for continuous learning"""
        try:
            dataset_dir = Path(self.config.get("dataset_dir", "captcha_dataset"))
            dataset_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            instance_id = f"INTEGRATION_{self.session_stats['total_attempts']:06d}"
            base_name = f"{solution.upper()}_{timestamp}_{instance_id}"
            
            # Copy image file
            img_dest = dataset_dir / f"{base_name}.png"
            shutil.copy2(image_path, img_dest)
            
            # Create JSON metadata
            metadata = {
                "solution": solution,
                "timestamp": datetime.now().isoformat(),
                "source": "phase1_integration",
                "original_path": str(Path(image_path).resolve()),
                "solver_confidence": "high",
                "integration_session": self.session_stats["session_start"],
                "attempt_number": self.session_stats["total_attempts"],
                "image_dimensions": "200x80"
            }
            
            json_dest = dataset_dir / f"{base_name}.json"
            with open(json_dest, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.session_stats["saved_to_dataset"] += 1
            logger.debug(f"Saved to dataset: {base_name}")
            
        except Exception as e:
            logger.warning(f"Failed to save to dataset: {e}")
    
    def _update_session_stats(self, result: str, solve_time: float):
        """Update session statistics"""
        if result == "success":
            self.session_stats["ml_successes"] += 1
        elif result == "failure":
            self.session_stats["failures"] += 1
        
        # Update average solve time
        total_attempts = self.session_stats["total_attempts"]
        current_avg = self.session_stats["avg_solve_time"]
        self.session_stats["avg_solve_time"] = (current_avg * (total_attempts - 1) + solve_time) / total_attempts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive solving statistics"""
        stats = self.session_stats.copy()
        
        # Calculate rates
        total = stats["total_attempts"]
        if total > 0:
            stats["success_rate"] = stats["ml_successes"] / total
            stats["failure_rate"] = stats["failures"] / total
            stats["api_fallback_rate"] = stats["api_fallbacks"] / total
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
            stats["api_fallback_rate"] = 0.0
        
        # Add solver statistics if available
        if self.solver:
            solver_stats = self.solver.get_performance_stats()
            stats["solver_stats"] = solver_stats
        
        return stats
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check of the integration"""
        health_status = {
            "solver_initialized": self.solver is not None,
            "models_loaded": False,
            "config_loaded": bool(self.config),
            "dataset_dir_accessible": False,
            "gpu_available": False
        }
        
        try:
            # Check models
            if self.solver:
                health_status["models_loaded"] = (
                    self.solver.segmentation_model is not None or 
                    self.solver.crnn_model is not None
                )
                health_status["gpu_available"] = self.solver.gpu_available
            
            # Check dataset directory
            dataset_dir = Path(self.config.get("dataset_dir", "captcha_dataset"))
            health_status["dataset_dir_accessible"] = dataset_dir.exists() or dataset_dir.parent.exists()
            
        except Exception as e:
            logger.warning(f"Health check error: {e}")
        
        return health_status
    
    def reset_session_stats(self):
        """Reset session statistics"""
        self.session_stats = {
            "session_start": datetime.now().isoformat(),
            "total_attempts": 0,
            "ml_successes": 0,
            "api_fallbacks": 0,
            "failures": 0,
            "avg_solve_time": 0.0,
            "saved_to_dataset": 0
        }
        logger.info("Session statistics reset")

# Global solver instance for easy integration
_global_solver_instance = None

def get_solver(model_dir: str = "models", config_file: str = "config/captcha_config.json", 
               api_key: str = None, force_reload: bool = False) -> ECICaptchaIntegration:
    """
    Get global solver instance (singleton pattern)
    
    Args:
        model_dir: Directory containing trained models
        config_file: Configuration file path
        api_key: Fallback API key
        force_reload: Force reload of solver instance
        
    Returns:
        ECICaptchaIntegration instance
    """
    global _global_solver_instance
    
    if _global_solver_instance is None or force_reload:
        logger.info("Initializing global CAPTCHA solver instance...")
        _global_solver_instance = ECICaptchaIntegration(
            model_dir=model_dir,
            config_file=config_file,
            fallback_api_key=api_key
        )
    
    return _global_solver_instance

def solve_captcha_image(image_path: str, confidence_threshold: float = 0.80,
                       save_to_dataset: bool = True, max_retries: int = 3) -> Optional[str]:
    """
    Simple function to solve CAPTCHA - drop-in replacement for AntiCaptcha
    
    Args:
        image_path: Path to CAPTCHA image
        confidence_threshold: Minimum confidence for ML prediction
        save_to_dataset: Whether to save solved CAPTCHA for learning
        max_retries: Maximum retry attempts
        
    Returns:
        CAPTCHA solution string or None if failed
    """
    try:
        solver = get_solver()
        return solver.solve_captcha(
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            max_retries=max_retries,
            save_to_dataset=save_to_dataset
        )
    except Exception as e:
        logger.error(f"solve_captcha_image failed: {e}")
        return None

def get_solver_statistics() -> Dict[str, Any]:
    """Get statistics from global solver instance"""
    try:
        solver = get_solver()
        return solver.get_statistics()
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {}

def perform_health_check() -> Dict[str, bool]:
    """Perform health check on global solver instance"""
    try:
        solver = get_solver()
        return solver.health_check()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"error": True}

# Integration Examples and Documentation
INTEGRATION_EXAMPLES = """
# PHASE 1 INTEGRATION EXAMPLES

## Method 1: Drop-in Replacement
Replace your existing AntiCaptcha calls:

# OLD CODE:
from anticaptcha import solve_captcha
solution = solve_captcha(image_path, api_key)

# NEW CODE:
from scripts.captcha_integration import solve_captcha_image
solution = solve_captcha_image(image_path)

## Method 2: Advanced Integration
For more control and monitoring:

from scripts.captcha_integration import ECICaptchaIntegration

# Initialize once
captcha_solver = ECICaptchaIntegration(
    model_dir="models",
    config_file="config/captcha_config.json",
    fallback_api_key="your_anticaptcha_key"  # Optional
)

# Use in download loop
async def download_with_captcha(page):
    # Save CAPTCHA image
    captcha_element = await page.query_selector("img[alt='captcha']")
    await captcha_element.screenshot(path="temp_captcha.png")
    
    # Solve CAPTCHA
    solution = captcha_solver.solve_captcha("temp_captcha.png")
    
    if solution:
        await page.fill("input[name='captcha']", solution)
        await page.click("button[type='submit']")
        return True
    else:
        logger.error("CAPTCHA solving failed")
        return False

## Method 3: Batch Processing
For processing multiple CAPTCHAs:

from scripts.captcha_integration import get_solver

solver = get_solver()

for captcha_file in captcha_files:
    solution = solver.solve_captcha(captcha_file)
    if solution:
        print(f"{captcha_file}: {solution}")

# Get performance statistics
stats = solver.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
"""

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ECI CAPTCHA Solver Integration")
    parser.add_argument("image_path", nargs="?", help="Path to CAPTCHA image")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--api-key", help="Fallback API key")
    parser.add_argument("--confidence", type=float, default=0.80, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save to dataset")
    parser.add_argument("--stats", action="store_true", help="Show solver statistics")
    parser.add_argument("--health", action="store_true", help="Perform health check")
    parser.add_argument("--examples", action="store_true", help="Show integration examples")
    
    args = parser.parse_args()
    
    if args.examples:
        print(INTEGRATION_EXAMPLES)
        return 0
    
    if args.health:
        health = perform_health_check()
        print("🏥 CAPTCHA Solver Health Check:")
        for check, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check}: {status}")
        return 0
    
    if args.stats:
        stats = get_solver_statistics()
        print("📊 CAPTCHA Solver Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        return 0
    
    if not args.image_path:
        parser.print_help()
        return 1
    
    # Solve single CAPTCHA
    print(f"🔍 Solving CAPTCHA: {args.image_path}")
    
    solver = ECICaptchaIntegration(
        model_dir=args.model_dir,
        config_file=args.config,
        fallback_api_key=args.api_key
    )
    
    start_time = time.time()
    solution = solver.solve_captcha(
        args.image_path,
        confidence_threshold=args.confidence,
        save_to_dataset=not args.no_save
    )
    solve_time = time.time() - start_time
    
    if solution:
        print(f"✅ Solution: {solution}")
        print(f"⏱️  Time: {solve_time:.2f}s")
        
        # Show statistics
        stats = solver.get_statistics()
        print(f"📈 Success rate: {stats['success_rate']:.1%}")
        
        return 0
    else:
        print("❌ Failed to solve CAPTCHA")
        return 1

if __name__ == "__main__":
    exit(main())
