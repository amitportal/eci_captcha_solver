#!/usr/bin/env python3
"""
Script to train models with all three preprocessing methods
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import subprocess
import os
from pathlib import Path

def train_model(preprocessing_method, dataset_dir, model_dir="models", csv_file=None):
    """Train a model with the specified preprocessing method"""
    print(f"Training model with {preprocessing_method} preprocessing...")

    # Always use .venv-captcha environment for training
    venv_python = os.path.join(".venv-captcha", "Scripts", "python.exe") if os.name == "nt" else os.path.join(".venv-captcha", "bin", "python")
    if not os.path.exists(venv_python):
        print(f"ERROR: .venv-captcha Python executable not found at {venv_python}")
        return False

    cmd = [
        venv_python, "main_solver.py",
        "--mode", "train",
        "--dataset-dir", dataset_dir,
        "--model-dir", model_dir,
        "--preprocessing-method", preprocessing_method
    ]

    if csv_file:
        cmd.extend(["--csv-file", csv_file])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"Training completed successfully for {preprocessing_method}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training model with {preprocessing_method}:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_all_models.py captcha_dataset captcha_solutions_consolidated.csv")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Preprocessing methods
    methods = ["grayscale", "light_smoothing", "precise_mathematical"]
    
    # Train models for each preprocessing method
    for method in methods:
        success = train_model(method, dataset_dir, csv_file=csv_file)
        if not success:
            print(f"Failed to train model with {method}")
            continue
    
    print("All models trained successfully!")

if __name__ == "__main__":
    main()