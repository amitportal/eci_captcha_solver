#!/usr/bin/env python3
"""
Example script showing how to use the solver with different preprocessing methods
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_solver import ProductionCaptchaSolver
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def solve_with_method(image_path, preprocessing_method):
    """Solve a CAPTCHA using a specific preprocessing method"""
    print(f"\n--- Solving with {preprocessing_method} preprocessing ---")
    
    # Create solver with specific preprocessing method
    solver = ProductionCaptchaSolver(
        model_dir="models",
        preprocessing_method=preprocessing_method
    )
    
    # Solve the CAPTCHA
    result = solver.solve(image_path)
    
    if result:
        print(f"✓ Solution: {result}")
    else:
        print("✗ Failed to solve CAPTCHA")
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python example_solver.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Try solving with all three preprocessing methods
    methods = ["grayscale", "light_smoothing", "precise_mathematical"]
    
    results = {}
    for method in methods:
        try:
            result = solve_with_method(image_path, method)
            results[method] = result
        except Exception as e:
            print(f"Error solving with {method}: {e}")
            results[method] = None
    
    # Show summary
    print("\n--- Summary ---")
    for method, result in results.items():
        status = result if result else "Failed"
        print(f"{method:20}: {status}")

if __name__ == "__main__":
    main()