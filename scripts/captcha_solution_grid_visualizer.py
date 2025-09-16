#!/usr/bin/env python3
"""
captcha_solution_grid_visualizer.py - Visualize CAPTCHA solutions in a grid format
Creates visual comparison grids showing original images with all solution variants
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CaptchaSolutionGridVisualizer:
    """Create visual comparison grids of CAPTCHA images with their solutions"""
    
    def __init__(self, csv_file: str, captcha_dir: str = "other_captchas", output_dir: str = "performance_results/solution_grids"):
        self.csv_file = Path(csv_file)
        self.captcha_dir = Path(captcha_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load performance results
        self.df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded {len(self.df)} CAPTCHA performance results")
        
        # Define solution columns to visualize
        self.solution_columns = [
            'original_solution',
            'grayscale_segment_solution',
            'grayscale_crnn_solution', 
            'light_smoothing_segment_solution',
            'light_smoothing_crnn_solution',
            'precise_mathematical_segment_solution',
            'precise_mathematical_crnn_solution',
            'raw_segment_solution',
            'raw_crnn_solution'
        ]
    
    def create_solution_grids(self, max_grids: Optional[int] = None, grid_size: Tuple[int, int] = (10, 5)) -> List[str]:
        """
        Create visualization grids comparing original images with all solution variants
        
        Args:
            max_grids: Maximum number of grids to create (None for all needed)
            grid_size: Grid dimensions (rows, columns) - defaults to 6x16
            
        Returns:
            List of created grid file paths
        """
        logger.info(f"Creating solution comparison grids...")
        logger.info(f"Grid size: {grid_size[0]} rows × {grid_size[1]} columns")
        
        rows, cols = grid_size
        captchas_per_grid = rows * cols
        total_grids_needed = (len(self.df) + captchas_per_grid - 1) // captchas_per_grid
        
        if max_grids is None:
            max_grids = total_grids_needed
        else:
            max_grids = min(max_grids, total_grids_needed)
        
        grid_files = []
        
        for grid_idx in range(max_grids):
            start_idx = grid_idx * captchas_per_grid
            end_idx = min(start_idx + captchas_per_grid, len(self.df))
            
            if start_idx >= len(self.df):
                break
                
            grid_data = self.df.iloc[start_idx:end_idx]
            grid_file = self._create_single_grid(grid_data, grid_idx + 1, rows, cols)
            if grid_file:
                grid_files.append(grid_file)
        
        logger.info(f"Created {len(grid_files)} solution comparison grids")
        return grid_files
    
    def _create_single_grid(self, data_df: pd.DataFrame, grid_num: int, rows: int, cols: int) -> Optional[str]:
        """
        Create a single visualization grid with CAPTCHA images and their solutions
        
        Args:
            data_df: DataFrame slice containing CAPTCHA data for this grid
            grid_num: Grid number for naming
            rows: Number of rows in grid
            cols: Number of columns in grid
            
        Returns:
            Path to created grid file or None if failed
        """
        try:
            # Create figure with subplots for the grid
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
            
            # Ensure axes is always 2D
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Process each CAPTCHA in the grid
            for idx, (_, row) in enumerate(data_df.iterrows()):
                if idx >= rows * cols:
                    break
                
                grid_row = idx // cols
                grid_col = idx % cols
                ax = axes[grid_row, grid_col]
                
                self._add_captcha_to_subplot(ax, row)
            
            # Hide unused subplots
            for idx in range(len(data_df), rows * cols):
                grid_row = idx // cols
                grid_col = idx % cols
                if grid_row < rows and grid_col < cols:
                    axes[grid_row, grid_col].axis('off')
                    axes[grid_row, grid_col].set_visible(False)
            
            # Add title to the entire figure
            fig.suptitle(f'CAPTCHA Solution Comparison Grid #{grid_num} - {len(data_df)} Samples', 
                         fontsize=16, y=0.98)
            
            # Adjust layout to prevent overlapping
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save grid
            output_file = self.output_dir / f"solution_comparison_grid_{grid_num}_{self.timestamp}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"✓ Created solution comparison grid #{grid_num}: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create solution comparison grid #{grid_num}: {e}")
            return None
    
    def _add_captcha_to_subplot(self, ax, row):
        """
        Add a single CAPTCHA image with its solutions to a subplot
        
        Args:
            ax: Matplotlib axis to draw on
            row: DataFrame row containing CAPTCHA data
        """
        try:
            filename = row['filename']
            img_path = self.captcha_dir / filename
            
            if img_path.exists():
                # Load and display image
                img = Image.open(img_path)
                ax.imshow(img, cmap='gray')
                
                # Create solution comparison text with colored indicators
                self._add_colored_solutions_to_subplot(ax, row)
                
                # Turn off axis ticks and labels for cleaner display
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(filename.split('_')[0], fontsize=10, pad=30)
            else:
                # Handle missing image
                ax.text(0.5, 0.5, f"Image not found:\n{filename}", 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=8, wrap=True)
                ax.set_title(filename.split('_')[0], fontsize=10, pad=30)
                ax.axis('off')
                
        except Exception as e:
            # Handle any other errors
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=8, wrap=True)
            ax.axis('off')
    
    def _add_colored_solutions_to_subplot(self, ax, row):
        """
        Add solution text with colored indicators to subplot
        
        Args:
            ax: Matplotlib axis to draw on
            row: DataFrame row containing CAPTCHA data
        """
        try:
            # Add original solution
            original = str(row.get('original_solution', 'N/A')).lower()
            ax.text(0.5, 0.05, f"Orig: {original}", 
                   transform=ax.transAxes, 
                   ha='center', va='bottom', 
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add ML solutions with colored indicators, one type per row
            ml_solutions = [
                ('GraySeg', 'grayscale_segment_solution', 'grayscale_segment_match'),
                ('GrayCRNN', 'grayscale_crnn_solution', 'grayscale_crnn_match'),
                ('LightSeg', 'light_smoothing_segment_solution', 'light_smoothing_segment_match'),
                ('LightCRNN', 'light_smoothing_crnn_solution', 'light_smoothing_crnn_match'),
                ('MathSeg', 'precise_mathematical_segment_solution', 'precise_mathematical_segment_match'),
                ('MathCRNN', 'precise_mathematical_crnn_solution', 'precise_mathematical_crnn_match'),
                ('RawSeg', 'raw_segment_solution', 'raw_segment_match'),
                ('RawCRNN', 'raw_crnn_solution', 'raw_crnn_match')
            ]
            
            # Position solutions in rows below the original solution
            # Each solution type gets its own row
            for i, (label, solution_col, match_col) in enumerate(ml_solutions):
                solution = str(row.get(solution_col, 'N/A')).lower()
                is_match = row.get(match_col, False)
                
                # Calculate vertical position for this solution row
                # Start below original solution (0.0) and space each row by 0.15 for better separation
                y_pos = 0.0 - (i * 0.15)
                
                # Display solution text with colored indicator
                indicator = "✓" if is_match else "✗"
                color = "green" if is_match else "red"
                
                # Combine label, solution, and indicator in one text element
                solution_text = f"{label}: {solution} {indicator}"
                ax.text(0.5, y_pos, solution_text,
                       transform=ax.transAxes,
                       ha='center', va='top',
                       fontsize=8, fontweight='bold' if is_match else 'normal',
                       color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
        except Exception as e:
            # Fallback text if coloring fails
            solution_text = self._format_solutions_for_display(row)
            ax.text(0.5, -2.0, solution_text, 
                   transform=ax.transAxes, 
                   ha='center', va='top', 
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _format_solutions_for_display(self, row):
        """
        Format all solutions for display as fallback text
        
        Args:
            row: DataFrame row containing CAPTCHA data
            
        Returns:
            Formatted string with all solutions
        """
        try:
            # Get all solution columns
            solution_columns = [
                'original_solution',
                'grayscale_segment_solution',
                'grayscale_crnn_solution', 
                'light_smoothing_segment_solution',
                'light_smoothing_crnn_solution',
                'precise_mathematical_segment_solution',
                'precise_mathematical_crnn_solution',
                'raw_segment_solution',
                'raw_crnn_solution'
            ]
            
            # Format solutions
            solutions = []
            for col in solution_columns:
                if col in row:
                    solution = str(row[col]).lower() if pd.notna(row[col]) else 'N/A'
                    # Extract short label from column name
                    label = col.replace('_solution', '').replace('_', ' ').title()
                    solutions.append(f"{label}: {solution}")
            
            return '\n'.join(solutions)
        except Exception:
            return "Error formatting solutions"
    
    

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Visualize CAPTCHA solutions in grid format for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s performance_results/performance_results_20250916_054737.csv
  %(prog)s --grid-size 4 20 --max-grids 5 performance_results/performance_results_20250916_054737.csv
  %(prog)s --captcha-dir other_captchas --output-dir grids performance_results/performance_results_20250916_054737.csv
        """
    )
    
    parser.add_argument("csv_file", help="Performance results CSV file")
    parser.add_argument("--captcha-dir", default="other_captchas", 
                       help="Directory containing CAPTCHA images (default: other_captchas)")
    parser.add_argument("--output-dir", default="performance_results/solution_grids",
                       help="Output directory for grid images (default: performance_results/solution_grids)")
    parser.add_argument("--max-grids", type=int, default=None,
                       help="Maximum number of grids to create (default: all needed)")
    parser.add_argument("--grid-size", nargs=2, type=int, default=[10, 5],
                       metavar=('ROWS', 'COLS'),
                       help="Grid dimensions as 'rows cols' (default: 10 5)")
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = CaptchaSolutionGridVisualizer(
            csv_file=args.csv_file,
            captcha_dir=args.captcha_dir,
            output_dir=args.output_dir
        )
        
        # Create solution grids
        grid_files = visualizer.create_solution_grids(
            max_grids=args.max_grids,
            grid_size=tuple(args.grid_size)
        )
        
        if grid_files:
            print(f"\n✅ Successfully created {len(grid_files)} solution comparison grids!")
            print(f"📁 Output directory: {args.output_dir}")
            print(f"📊 Grid dimensions: {args.grid_size[0]}×{args.grid_size[1]} ({args.grid_size[0]*args.grid_size[1]} CAPTCHAs per grid)")
            
            for i, grid_file in enumerate(grid_files, 1):
                print(f"  {i:2d}. {os.path.basename(grid_file)}")
        else:
            print("❌ No solution comparison grids were created.")
            
    except Exception as e:
        logger.error(f"Failed to create solution comparison grids: {e}")
        raise

if __name__ == "__main__":
    main()