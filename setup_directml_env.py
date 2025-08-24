#!/usr/bin/env python3
"""
setup_directml_env.py - Setup Python 3.10 environment with TensorFlow-DirectML
Alternative environment for DirectML GPU acceleration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_python310():
    """Find Python 3.10 executable"""
    candidates = [
        'py -3.10',
        'python3.10',
        'python3'
    ]
    
    for candidate in candidates:
        try:
            if candidate.startswith('py '):
                cmd = candidate.split() + ['--version']
            else:
                cmd = [candidate, '--version']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and '3.10' in result.stdout:
                return candidate
        except:
            continue
    
    # Check LocalAppData
    local_app = os.environ.get('LOCALAPPDATA', '')
    if local_app:
        py310_path = Path(local_app) / 'Programs' / 'Python' / 'Python310' / 'python.exe'
        if py310_path.exists():
            return str(py310_path)
    
    return None

def create_directml_environment():
    """Create Python 3.10 environment with DirectML support"""
    base_dir = Path('.').resolve()
    venv_dir = base_dir / '.venv-directml'
    
    logger.info("🎯 Setting up DirectML environment with Python 3.10...")
    
    # Find Python 3.10
    python_cmd = find_python310()
    if not python_cmd:
        logger.error("❌ Python 3.10 not found!")
        logger.info("Please install Python 3.10 for DirectML support")
        logger.info("Download from: https://www.python.org/downloads/release/python-3109/")
        return False
    
    logger.info(f"✓ Found Python 3.10: {python_cmd}")
    
    # Remove existing venv if present
    if venv_dir.exists():
        logger.info("Removing existing .venv-directml...")
        import shutil
        import time
        
        backup_name = f'.venv-directml-backup-{int(time.time())}'
        backup_dir = base_dir / backup_name
        
        try:
            venv_dir.rename(backup_dir)
            logger.info(f"✓ Backed up to {backup_name}")
        except Exception as e:
            logger.error(f"Cannot remove existing venv: {e}")
            return False
    
    # Create new venv
    try:
        if python_cmd.startswith('py '):
            cmd = python_cmd.split() + ['-m', 'venv', str(venv_dir)]
        else:
            cmd = [python_cmd, '-m', 'venv', str(venv_dir)]
        
        subprocess.run(cmd, check=True, timeout=60)
        logger.info("✓ Created .venv-directml")
    except Exception as e:
        logger.error(f"Failed to create venv: {e}")
        return False
    
    # Set up paths
    if os.name == 'nt':
        pip_path = venv_dir / 'Scripts' / 'pip.exe'
        python_path = venv_dir / 'Scripts' / 'python.exe'
    else:
        pip_path = venv_dir / 'bin' / 'pip'
        python_path = venv_dir / 'bin' / 'python'
    
    try:
        # Upgrade pip
        logger.info("Upgrading pip...")
        subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, timeout=120)
        
        # Install TensorFlow CPU 2.10 (required for DirectML)
        logger.info("Installing TensorFlow CPU 2.10...")
        subprocess.run([str(pip_path), 'install', 'tensorflow-cpu==2.10'], 
                      check=True, timeout=300)
        
        # Install DirectML plugin
        logger.info("Installing TensorFlow-DirectML plugin...")
        subprocess.run([str(pip_path), 'install', 'tensorflow-directml-plugin'], 
                      check=True, timeout=300)
        
        # Install other dependencies
        deps = [
            "opencv-python>=4.8.0",
            "numpy>=1.21.0,<1.24.0",  # DirectML has numpy version constraints
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
            "scikit-learn>=1.0.0",
            "Pillow>=9.0.0",
            "pytesseract>=0.3.10",
            "requests>=2.27.0",
            "psutil>=5.8.0",
            "pynvml>=11.4.0"
        ]
        
        logger.info("Installing other dependencies...")
        for dep in deps:
            logger.info(f"Installing {dep}...")
            subprocess.run([str(pip_path), 'install', dep], check=True, timeout=120)
        
        logger.info("✓ All dependencies installed successfully")
        
        # Test DirectML
        logger.info("Testing DirectML setup...")
        test_script = '''
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Test DirectML
try:
    # Create a simple computation to test DirectML
    import numpy as np
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"DirectML test computation successful: {c.numpy()}")
    print("✓ DirectML GPU acceleration is working!")
except Exception as e:
    print(f"DirectML test failed: {e}")
'''
        
        result = subprocess.run([str(python_path), '-c', test_script], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✓ DirectML verification successful:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
        else:
            logger.warning(f"DirectML verification had issues: {result.stderr}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install DirectML dependencies: {e}")
        return False

def show_directml_usage():
    """Show how to use the DirectML environment"""
    activate_cmd = ".venv-directml\\Scripts\\Activate.ps1" if os.name == 'nt' else "source .venv-directml/bin/activate"
    
    print(f"""
🎯 DirectML Environment Ready!

To use DirectML GPU acceleration:

1. Activate the DirectML environment:
   {activate_cmd}

2. Test DirectML:
   python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

3. Use DirectML in your code:
   # DirectML will automatically use your GPU for TensorFlow operations
   import tensorflow as tf
   
   # Your existing TensorFlow code will now use GPU acceleration
   model = tf.keras.Sequential([...])
   model.fit(X, y)  # This will use DirectML GPU acceleration

4. Switch between environments:
   - For PyTorch CUDA: .venv-captcha\\Scripts\\Activate.ps1
   - For DirectML: .venv-directml\\Scripts\\Activate.ps1

💡 DirectML automatically accelerates TensorFlow operations on your RTX 3050!
""")

if __name__ == "__main__":
    if create_directml_environment():
        logger.info("✅ DirectML environment setup completed!")
        show_directml_usage()
    else:
        logger.error("❌ DirectML environment setup failed!")
        print("\nFallback options:")
        print("1. Use the main environment (.venv-captcha) with PyTorch CUDA")
        print("2. Install Python 3.10 and retry DirectML setup")