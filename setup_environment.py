#!/usr/bin/env python3
"""
setup_environment.py - Complete Environment Setup for ECI CAPTCHA Solver
Creates directory structure, installs dependencies, and configures environment
"""

import os
import sys
import io
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure UTF-8 encoding for console output
try:
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

class EnvironmentSetup:
    """Complete environment setup and configuration"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()
        logger.info(f"Setting up environment in: {self.base_dir}")
        
    def create_directory_structure(self):
        """Create complete directory structure"""
        directories = {
            'captcha_dataset': "Training data (PNG/JSON files)",
            'models': "Trained models",
            'config': "Configuration files", 
            'scripts': "Utility scripts",
            'logs': "Training and inference logs",
            'output': "Test results and reports"
        }
        
        logger.info("Creating directory structure...")
        
        for dir_name, description in directories.items():
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created {dir_name}/ - {description}")
            else:
                logger.info(f"✓ Directory {dir_name}/ already exists")
        
        return True

    def find_python311(self) -> str:
        """Find Python 3.11 executable"""
        # Try py launcher first
        try:
            result = subprocess.run(['py', '-3.11', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and '3.11' in result.stdout:
                return 'py -3.11'
        except:
            pass
        
        # Try LocalAppData default location
        local_app = os.environ.get('LOCALAPPDATA', '')
        if local_app:
            py311_path = Path(local_app) / 'Programs' / 'Python' / 'Python311' / 'python.exe'
            if py311_path.exists():
                return str(py311_path)
        
        # Try PATH candidates
        for candidate in ['python3.11', 'python3', 'python']:
            try:
                result = subprocess.run([candidate, '--version'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and '3.11' in result.stdout:
                    return candidate
            except:
                continue
        
        logger.error("Python 3.11 not found! Please install Python 3.11")
        return None

    def create_venv_and_install(self):
        """Create fresh venv and install all dependencies including TensorFlow"""
        venv_dir = self.base_dir / '.venv-captcha'
        
        # Remove existing venv if present
        if venv_dir.exists():
            logger.info("Removing existing .venv-captcha...")
            import shutil
            import time
            
            # Try to rename first (safer on Windows), then delete
            backup_name = f'.venv-captcha-backup-{int(time.time())}'
            backup_dir = self.base_dir / backup_name
            
            try:
                # Try to rename the old venv first
                venv_dir.rename(backup_dir)
                logger.info(f"✓ Renamed existing .venv-captcha to {backup_name}")
                
                # Now try to delete the backup
                try:
                    shutil.rmtree(backup_dir)
                    logger.info("✓ Cleaned up old venv")
                except:
                    logger.warning(f"Old venv backed up as {backup_name} (couldn't delete)")
                    
            except Exception as e:
                logger.error(f"Cannot rename/remove existing .venv-captcha: {e}")
                logger.info("Please manually delete or rename the .venv-captcha folder and run again.")
                return False
        
        # Find Python 3.11
        python_cmd = self.find_python311()
        if not python_cmd:
            return False
        
        logger.info(f"Creating fresh .venv-captcha using {python_cmd}...")
        
        # Create new venv
        try:
            if python_cmd.startswith('py '):
                cmd = python_cmd.split() + ['-m', 'venv', str(venv_dir)]
            else:
                cmd = [python_cmd, '-m', 'venv', str(venv_dir)]
            
            subprocess.run(cmd, check=True, timeout=60)
            logger.info("✓ Created .venv-captcha")
        except Exception as e:
            logger.error(f"Failed to create venv: {e}")
            return False
        
        # Install dependencies
        if os.name == 'nt':
            pip_path = venv_dir / 'Scripts' / 'pip.exe'
            python_path = venv_dir / 'Scripts' / 'python.exe'
        else:
            pip_path = venv_dir / 'bin' / 'pip'
            python_path = venv_dir / 'bin' / 'python'
        
        try:
            # Upgrade pip first
            logger.info("Upgrading pip...")
            subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True, timeout=120)
            
            # Install TensorFlow (CPU version for now)
            logger.info("Installing TensorFlow...")
            subprocess.run([str(pip_path), 'install', 'tensorflow'], check=True, timeout=300)
            
            # Install other dependencies
            dependencies = [
                "opencv-python>=4.8.0",
                "pandas>=2.0.0",
                "matplotlib>=3.7.0",
                "scikit-learn>=1.3.0",
                "Pillow>=10.0.0",
                "pytesseract>=0.3.10",
                "requests>=2.31.0",
                "psutil>=5.9.0",
                "pynvml>=11.4.0"
            ]
            
            logger.info("Installing other dependencies...")
            for dep in dependencies:
                logger.info(f"Installing {dep}...")
                subprocess.run([str(pip_path), 'install', dep], check=True, timeout=120)
            
            # Install PyTorch with CUDA for GPU acceleration
            logger.info("Installing PyTorch with CUDA for GPU acceleration...")
            subprocess.run([str(pip_path), 'install', 'torch', 'torchvision', 'torchaudio', 
                           '--index-url', 'https://download.pytorch.org/whl/cu118'], check=True, timeout=600)
            
            logger.info("✓ All dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def verify_gpu_setup(self):
        """Verify GPU setup with TensorFlow and NVIDIA detection"""
        venv_dir = self.base_dir / '.venv-captcha'
        if os.name == 'nt':
            python_path = venv_dir / 'Scripts' / 'python.exe'
        else:
            python_path = venv_dir / 'bin' / 'python'
        
        if not python_path.exists():
            logger.error("Venv not found!")
            return False
        
        test_script = '''
import tensorflow as tf
import sys
print(f"Python: {sys.executable}")
print(f"TensorFlow version: {tf.__version__}")
print(f"TF GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"TF built with CUDA: {tf.test.is_built_with_cuda()}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch GPU count: {torch.cuda.device_count()}")
        print(f"PyTorch GPU 0: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")

try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    print(f"NVIDIA GPUs detected by pynvml: {count}")
    if count > 0:
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            print(f"GPU {i}: {name}")
    pynvml.nvmlShutdown()
except Exception as e:
    print(f"pynvml check failed: {e}")
'''
        
        try:
            logger.info("Verifying GPU setup...")
            result = subprocess.run([str(python_path), '-c', test_script], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✓ Environment verification results:")
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
                
                # Check if we have GPU available through any method
                if 'PyTorch CUDA available: True' in result.stdout:
                    logger.info("✓ GPU acceleration available via PyTorch CUDA!")
                elif 'TF GPU devices: []' not in result.stdout and 'GPU' in result.stdout:
                    logger.info("✓ GPU acceleration available via TensorFlow!")
                elif 'NVIDIA GPUs detected by pynvml: 0' in result.stdout:
                    logger.warning("⚠ No NVIDIA GPUs detected - will run on CPU only")
                else:
                    logger.info("ℹ NVIDIA GPU detected by drivers but not accessible to ML frameworks")
                    logger.info("  Consider installing CUDA toolkit or using WSL2 for better GPU support")
                
                return True
            else:
                logger.error(f"Environment verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Environment verification error: {e}")
            return False

    def create_configuration_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        config_dir = self.base_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Main configuration
        main_config = {
            "dataset_dir": "captcha_dataset",
            "model_dir": "models",
            "log_dir": "logs",
            "confidence_threshold": 0.80,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.0005,
            "img_width": 200,
            "img_height": 80,
            "captcha_length": 6,
            "use_gpu_training": True,
            "use_gpu_inference": "auto"
        }
        
        config_file = config_dir / "captcha_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(main_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Created config: {config_file}")
        return True

    def create_requirements_txt(self):
        """Create requirements.txt"""
        requirements = [
            "tensorflow>=2.20.0",
            "torch>=2.7.0",
            "torchvision>=0.22.0", 
            "torchaudio>=2.7.0",
            "opencv-python>=4.8.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "scikit-learn>=1.3.0",
            "Pillow>=10.0.0",
            "pytesseract>=0.3.10",
            "requests>=2.31.0",
            "psutil>=5.9.0",
            "pynvml>=11.4.0"
        ]
        
        req_file = self.base_dir / "requirements.txt"
        with open(req_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(requirements))
            f.write('\n\n# For PyTorch CUDA installation:\n')
            f.write('# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n')
        
        logger.info(f"✓ Created requirements.txt")
        return True

    def run_complete_setup(self):
        """Run complete setup"""
        logger.info("🚀 Starting complete ECI CAPTCHA solver setup...")
        
        success = True
        
        # 1. Create directories
        if not self.create_directory_structure():
            success = False
        
        # 2. Create fresh venv and install all dependencies
        if not self.create_venv_and_install():
            success = False
        
        # 3. Create config files
        if not self.create_configuration_files():
            success = False
        
        # 4. Create requirements.txt
        if not self.create_requirements_txt():
            success = False
        
        # 5. Verify GPU setup
        if not self.verify_gpu_setup():
            logger.warning("GPU verification failed, but setup may still work")
        
        if success:
            logger.info("✅ Setup completed successfully!")
            self.show_next_steps()
        else:
            logger.error("❌ Setup failed! Check logs for details.")
        
        return success

    def show_next_steps(self):
        """Show next steps to user"""
        venv_activate = ".venv-captcha\\Scripts\\Activate.ps1" if os.name == 'nt' else "source .venv-captcha/bin/activate"
        
        print(f"""
🎯 Setup Complete! Next Steps:

1. Activate the virtual environment:
   {venv_activate}

2. Test that everything works:
   python -c "import tensorflow as tf; print('TF version:', tf.__version__)"

3. Run the main solver:
   python main.py --mode test --dataset-dir captcha_dataset

4. Train models (if you have training data):
   python main.py --mode train --dataset-dir captcha_dataset --epochs 10

5. Solve a single captcha:
   python main.py --mode solve --image-path path/to/captcha.png

📋 What's installed:
   ✓ Python 3.11 virtual environment (.venv-captcha)
   ✓ TensorFlow 2.20.0 (CPU version)
   ✓ OpenCV, NumPy, Pandas, Scikit-learn
   ✓ NVIDIA GPU detection via pynvml

💡 For GPU acceleration:
   - Install CUDA toolkit for native GPU support, or
   - Use WSL2 with tensorflow[and-cuda] for better GPU support

Your environment is ready to start developing! 🚀
""")

        # Show specific GPU status
        try:
            venv_dir = self.base_dir / '.venv-captcha'
            python_path = venv_dir / 'Scripts' / 'python.exe' if os.name == 'nt' else venv_dir / 'bin' / 'python'
            
            result = subprocess.run([str(python_path), '-c', 
                "import pynvml; pynvml.nvmlInit(); print(f'GPU Status: {pynvml.nvmlDeviceGetCount()} NVIDIA GPU(s) detected')"], 
                capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"🔧 {result.stdout.strip()}")
        except:
            pass

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="ECI CAPTCHA Solver Environment Setup")
    parser.add_argument("--base-dir", default=".", help="Base directory for setup")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(args.base_dir)
    setup.run_complete_setup()

if __name__ == "__main__":
    main()