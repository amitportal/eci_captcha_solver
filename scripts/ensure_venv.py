#!/usr/bin/env python3
"""
scripts/ensure_venv.py - Reusable Virtual Environment Helper
Centralized venv detection, activation, and re-execution logic
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

def find_venv_path(venv_name: str = '.venv-captcha', start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Search upward from start_dir to find the virtual environment directory.
    
    Args:
        venv_name: Name of the virtual environment directory
        start_dir: Directory to start searching from (defaults to current working directory)
        
    Returns:
        Path to the virtual environment if found, None otherwise
    """
    if start_dir is None:
        start_dir = Path.cwd()
    
    current_dir = Path(start_dir).resolve()
    
    # Search upward through parent directories
    while current_dir != current_dir.parent:
        venv_path = current_dir / venv_name
        if venv_path.exists() and venv_path.is_dir():
            # Check if it's a valid virtual environment
            if os.name == 'nt':  # Windows
                python_exe = venv_path / 'Scripts' / 'python.exe'
            else:  # Unix-like
                python_exe = venv_path / 'bin' / 'python'
            
            if python_exe.exists():
                return venv_path
        
        current_dir = current_dir.parent
    
    return None

def get_venv_python_executable(venv_path: Path) -> Path:
    """
    Get the Python executable path for a virtual environment.
    
    Args:
        venv_path: Path to the virtual environment directory
        
    Returns:
        Path to the Python executable
    """
    if os.name == 'nt':  # Windows
        return venv_path / 'Scripts' / 'python.exe'
    else:  # Unix-like
        return venv_path / 'bin' / 'python'

def is_running_in_venv(venv_name: str = '.venv-captcha') -> Tuple[bool, Optional[Path]]:
    """
    Check if the current Python interpreter is running inside the specified virtual environment.
    
    Args:
        venv_name: Name of the virtual environment to check for
        
    Returns:
        Tuple of (is_in_venv, venv_path)
    """
    current_python = Path(sys.executable)
    venv_path = find_venv_path(venv_name)
    
    if venv_path is None:
        return False, None
    
    venv_python = get_venv_python_executable(venv_path)
    
    try:
        # Compare resolved paths to handle symlinks and relative paths
        return current_python.resolve() == venv_python.resolve(), venv_path
    except (OSError, ValueError):
        # Fallback to string comparison if resolve() fails
        return str(current_python).lower() == str(venv_python).lower(), venv_path

def ensure_venv_and_reexec(venv_name: str = '.venv-captcha', script_args: Optional[list] = None) -> bool:
    """
    Ensure the script is running in the specified virtual environment.
    If not, re-execute the script with the virtual environment's Python interpreter.
    
    Args:
        venv_name: Name of the virtual environment directory
        script_args: Arguments to pass to the re-executed script (defaults to sys.argv)
        
    Returns:
        True if already running in venv, False if re-execution was attempted
    """
    if script_args is None:
        script_args = sys.argv
    
    is_in_venv, venv_path = is_running_in_venv(venv_name)
    
    if is_in_venv:
        print(f"✓ Already running in {venv_name}")
        return True
    
    if venv_path is None:
        print(f"❌ Virtual environment '{venv_name}' not found")
        print("Please run setup_environment.py first to create the virtual environment")
        sys.exit(1)
    
    venv_python = get_venv_python_executable(venv_path)
    
    if not venv_python.exists():
        print(f"❌ Python executable not found in {venv_path}")
        sys.exit(1)
    
    print(f"🔄 Re-executing with {venv_name} Python interpreter...")
    print(f"   {venv_python}")
    
    try:
        # Re-execute the script with the virtual environment's Python
        result = subprocess.run([str(venv_python)] + script_args, check=False)
        sys.exit(result.returncode)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"❌ Failed to re-execute script: {e}")
        sys.exit(1)

def get_venv_info(venv_name: str = '.venv-captcha') -> dict:
    """
    Get information about the virtual environment.
    
    Args:
        venv_name: Name of the virtual environment directory
        
    Returns:
        Dictionary with virtual environment information
    """
    is_in_venv, venv_path = is_running_in_venv(venv_name)
    
    info = {
        'venv_name': venv_name,
        'is_in_venv': is_in_venv,
        'venv_path': str(venv_path) if venv_path else None,
        'current_python': sys.executable,
        'venv_python': None
    }
    
    if venv_path:
        venv_python = get_venv_python_executable(venv_path)
        info['venv_python'] = str(venv_python)
        info['venv_python_exists'] = venv_python.exists()
    
    return info

def print_venv_status(venv_name: str = '.venv-captcha'):
    """Print current virtual environment status."""
    info = get_venv_info(venv_name)
    
    print("\n🔧 Virtual Environment Status:")
    print(f"  Target venv: {info['venv_name']}")
    print(f"  Running in venv: {'✓' if info['is_in_venv'] else '❌'}")
    print(f"  Current Python: {info['current_python']}")
    
    if info['venv_path']:
        print(f"  Venv path: {info['venv_path']}")
        print(f"  Venv Python: {info['venv_python']}")
        print(f"  Venv Python exists: {'✓' if info.get('venv_python_exists') else '❌'}")
    else:
        print(f"  ❌ Virtual environment '{venv_name}' not found")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Virtual Environment Helper")
    parser.add_argument("--venv-name", default=".venv-captcha", help="Virtual environment name")
    parser.add_argument("--status", action="store_true", help="Show virtual environment status")
    parser.add_argument("--ensure", action="store_true", help="Ensure running in virtual environment")
    parser.add_argument("--find", action="store_true", help="Find virtual environment path")
    
    args = parser.parse_args()
    
    if args.status:
        print_venv_status(args.venv_name)
    elif args.find:
        venv_path = find_venv_path(args.venv_name)
        if venv_path:
            print(f"✓ Found {args.venv_name}: {venv_path}")
        else:
            print(f"❌ Virtual environment '{args.venv_name}' not found")
    elif args.ensure:
        ensure_venv_and_reexec(args.venv_name)
        print(f"✓ Running in {args.venv_name}")
    else:
        print_venv_status(args.venv_name)