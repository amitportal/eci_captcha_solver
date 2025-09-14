#!/usr/bin/env python3
"""
Simple CAPTCHA smoothing script

Applies median blur smoothing to CAPTCHA images.
"""

import cv2
import glob
import os

def smooth_captcha_images():
    """Apply smoothing to all CAPTCHA images"""
    
    # Find all CAPTCHA files
    captcha_files = glob.glob("test_captcha*.png")
    
    print(f"Found {len(captcha_files)} files to smooth")
    
    # Apply smoothing to each file
    for file in captcha_files:
        # Read the image
        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Could not read {file}")
            continue
            
        # Apply median blur smoothing
        smoothed = cv2.medianBlur(img, 3)
        
        # Create output filename
        output_file = file.replace(".png", "_smoothed.png")
        
        # Save the smoothed image
        success = cv2.imwrite(output_file, smoothed)
        
        if success:
            print(f"Smoothed {file} -> {output_file}")
        else:
            print(f"Failed to save {output_file}")
    
    print("Smoothing complete!")

if __name__ == "__main__":
    smooth_captcha_images()