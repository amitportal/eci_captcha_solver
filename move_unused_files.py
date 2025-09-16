#!/usr/bin/env python3
"""
Script to move unused CAPTCHA files to a separate directory
Moves all files in captcha_dataset that are not listed in captcha_solutions_consolidated.csv
"""

import os
import pandas as pd
import shutil
from pathlib import Path

def move_unused_files():
    # Define paths
    dataset_dir = Path("captcha_dataset")
    csv_file = "captcha_solutions_consolidated.csv"
    destination_dir = Path("other_captchas")
    
    # Create destination directory if it doesn't exist
    destination_dir.mkdir(exist_ok=True)
    
    # Read the full CSV file
    df = pd.read_csv(csv_file)
    
    # Get the set of filenames from the CSV
    csv_filenames = set()
    for _, row in df.iterrows():
        # Add the main filename
        csv_filenames.add(row['filename'])
        # Add the json_file filename if it exists
        if 'json_file' in row and pd.notna(row['json_file']):
            csv_filenames.add(row['json_file'])
        # Add the source_json filename if it exists
        if 'source_json' in row and pd.notna(row['source_json']):
            # Convert source_json to filename if it's a path
            source_json = str(row['source_json'])
            if source_json.endswith('.json'):
                csv_filenames.add(os.path.basename(source_json))
            else:
                # If it doesn't end with .json, assume it's a base name
                csv_filenames.add(f"{source_json}.json")
                csv_filenames.add(f"{source_json}.png")
    
    print(f"Number of unique files referenced in CSV: {len(csv_filenames)}")
    
    # Get all files in the dataset directory
    all_files = set()
    for file_path in dataset_dir.glob("*"):
        if file_path.is_file():
            all_files.add(file_path.name)
    
    print(f"Total files in dataset directory: {len(all_files)}")
    
    # Find files that are not in the CSV
    unused_files = all_files - csv_filenames
    print(f"Number of unused files: {len(unused_files)}")
    
    # Move unused files to the destination directory
    moved_count = 0
    for filename in unused_files:
        source_path = dataset_dir / filename
        destination_path = destination_dir / filename
        
        if source_path.exists():
            try:
                shutil.move(str(source_path), str(destination_path))
                moved_count += 1
                if moved_count <= 10:  # Show first 10 moves
                    print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    
    print(f"Moved {moved_count} files to {destination_dir}")
    
    # Also check for corresponding .png files for .json files
    additional_moves = 0
    for filename in list(unused_files):
        if filename.endswith('.json'):
            png_filename = filename[:-5] + '.png'  # Replace .json with .png
            png_source_path = dataset_dir / png_filename
            png_destination_path = destination_dir / png_filename
            
            if png_source_path.exists() and png_source_path.name not in csv_filenames:
                try:
                    shutil.move(str(png_source_path), str(png_destination_path))
                    additional_moves += 1
                    if additional_moves <= 10:  # Show first 10 additional moves
                        print(f"Also moved corresponding PNG: {png_filename}")
                except Exception as e:
                    print(f"Error moving corresponding PNG {png_filename}: {e}")
    
    if additional_moves > 0:
        print(f"Moved {additional_moves} additional PNG files")
    
    print("Done!")

if __name__ == "__main__":
    move_unused_files()