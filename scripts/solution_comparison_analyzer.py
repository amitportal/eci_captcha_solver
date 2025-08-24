#!/usr/bin/env python3
"""
solution_comparison_analyzer.py - Compare ML solutions with original solutions
Creates comprehensive analysis of model accuracy vs ground truth
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
from collections import Counter
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolutionComparisonAnalyzer:
    """Analyze and compare ML solutions with original ground truth solutions"""
    
    def __init__(self, original_csv: str, ml_solutions_dir: str, output_dir: str = "output"):
        self.original_csv = Path(original_csv)
        self.ml_solutions_dir = Path(ml_solutions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load original solutions
        self.original_df = pd.read_csv(self.original_csv)
        logger.info(f"Loaded {len(self.original_df)} original solutions")
        
        # Load ML solutions
        self.ml_solutions = self._load_ml_solutions()
        logger.info(f"Loaded {len(self.ml_solutions)} ML solutions")
    
    def _load_ml_solutions(self) -> Dict[str, str]:
        """Load ML solutions from JSON files"""
        ml_solutions = {}
        
        # Find the latest batch directory
        batch_dirs = [d for d in self.ml_solutions_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        if not batch_dirs:
            logger.error(f"No batch directories found in {self.ml_solutions_dir}")
            return {}
        
        latest_batch = max(batch_dirs, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using latest batch: {latest_batch}")
        
        # Load all JSON solution files
        json_files = list(latest_batch.glob("*_solution.json"))
        logger.info(f"Found {len(json_files)} ML solution files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract filename and solution from the ML solver format
                if 'filename' in data and 'solution' in data and data.get('success', False):
                    filename = data['filename']
                    solution = data['solution']
                    ml_solutions[filename] = solution
                    
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return ml_solutions
    
    def create_merged_comparison_csv(self) -> str:
        """Create merged CSV with original and ML solutions"""
        logger.info("Creating merged comparison CSV...")
        
        # Add ML solution column
        self.original_df['ml_solution'] = self.original_df['filename'].map(self.ml_solutions)
        
        # Create comparison metrics
        self.original_df['has_ml_solution'] = self.original_df['ml_solution'].notna()
        self.original_df['solutions_match'] = (
            self.original_df['solution'].str.lower() == 
            self.original_df['ml_solution'].str.lower()
        )
        
        # Handle cases where ML solution is missing
        self.original_df['solutions_match'] = self.original_df['solutions_match'].fillna(False)
        
        # Calculate character-level accuracy
        self.original_df['char_accuracy'] = self.original_df.apply(
            lambda row: self._calculate_char_accuracy(row['solution'], row['ml_solution']) 
            if pd.notna(row['ml_solution']) else 0.0, axis=1
        )
        
        # Save merged CSV
        output_file = self.output_dir / f"captcha_solutions_comparison_{self.timestamp}.csv"
        self.original_df.to_csv(output_file, index=False)
        logger.info(f"Merged comparison saved to: {output_file}")
        
        return str(output_file)
    
    def _calculate_char_accuracy(self, original: str, ml_solution: str) -> float:
        """Calculate character-level accuracy between two solutions"""
        if not original or not ml_solution:
            return 0.0
        
        original = str(original).lower()
        ml_solution = str(ml_solution).lower()
        
        # Pad shorter string with spaces
        max_len = max(len(original), len(ml_solution))
        original = original.ljust(max_len)
        ml_solution = ml_solution.ljust(max_len)
        
        correct_chars = sum(1 for o, m in zip(original, ml_solution) if o == m)
        return correct_chars / max_len
    
    def create_differences_csv(self) -> str:
        """Create CSV with only mismatched solutions"""
        logger.info("Creating differences CSV...")
        
        # Filter for mismatches
        differences_df = self.original_df[
            self.original_df['has_ml_solution'] & 
            (~self.original_df['solutions_match'])
        ].copy()
        
        # Add analysis columns
        differences_df['original_length'] = differences_df['solution'].str.len()
        differences_df['ml_length'] = differences_df['ml_solution'].str.len()
        differences_df['length_diff'] = differences_df['ml_length'] - differences_df['original_length']
        
        # Analyze common error patterns
        differences_df['error_type'] = differences_df.apply(self._classify_error, axis=1)
        
        # Save differences CSV
        output_file = self.output_dir / f"captcha_solutions_differences_{self.timestamp}.csv"
        differences_df.to_csv(output_file, index=False)
        logger.info(f"Differences CSV saved to: {output_file}")
        logger.info(f"Found {len(differences_df)} mismatched solutions")
        
        return str(output_file)
    
    def _classify_error(self, row) -> str:
        """Classify the type of error"""
        original = str(row['solution']).lower()
        ml_solution = str(row['ml_solution']).lower()
        
        if len(original) != len(ml_solution):
            return "length_mismatch"
        elif original.isdigit() and not ml_solution.isdigit():
            return "digit_to_char"
        elif not original.isdigit() and ml_solution.isdigit():
            return "char_to_digit"
        elif abs(ord(original[0]) - ord(ml_solution[0])) <= 2:
            return "similar_chars"
        else:
            return "different_chars"
    
    def _analyze_character_frequencies(self) -> Dict:
        """Analyze character frequencies in the dataset"""
        import string
        
        # Collect all characters from solutions
        all_chars = ''.join(self.original_df['solution'].astype(str).str.lower())
        
        # Count character frequencies
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        # Categorize characters
        digits = string.digits
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        
        digit_counts = {char: count for char, count in char_counts.items() if char in digits}
        lowercase_counts = {char: count for char, count in char_counts.items() if char in lowercase}
        uppercase_counts = {char: count for char, count in char_counts.items() if char in uppercase}
        other_counts = {char: count for char, count in char_counts.items() 
                       if char not in digits and char not in lowercase and char not in uppercase}
        
        # Calculate percentages and statistics
        digit_total = sum(digit_counts.values())
        lowercase_total = sum(lowercase_counts.values())
        uppercase_total = sum(uppercase_counts.values())
        other_total = sum(other_counts.values())
        
        return {
            'total_characters': total_chars,
            'digit_stats': {
                'count': digit_total,
                'percentage': (digit_total / total_chars * 100) if total_chars > 0 else 0,
                'frequency_by_digit': {k: {'count': v, 'percentage': v/total_chars*100} 
                                     for k, v in sorted(digit_counts.items())}
            },
            'lowercase_stats': {
                'count': lowercase_total,
                'percentage': (lowercase_total / total_chars * 100) if total_chars > 0 else 0,
                'frequency_by_letter': {k: {'count': v, 'percentage': v/total_chars*100} 
                                      for k, v in sorted(lowercase_counts.items())}
            },
            'uppercase_stats': {
                'count': uppercase_total,
                'percentage': (uppercase_total / total_chars * 100) if total_chars > 0 else 0,
                'frequency_by_letter': {k: {'count': v, 'percentage': v/total_chars*100} 
                                      for k, v in sorted(uppercase_counts.items())}
            },
            'other_stats': {
                'count': other_total,
                'percentage': (other_total / total_chars * 100) if total_chars > 0 else 0,
                'characters': other_counts
            }
        }
    
    def generate_statistics_report(self) -> str:
        """Generate comprehensive statistics report"""
        logger.info("Generating statistics report...")
        
        stats = {}
        
        # Basic accuracy metrics
        total_original = len(self.original_df)
        has_ml_solution = self.original_df['has_ml_solution'].sum()
        exact_matches = self.original_df['solutions_match'].sum()
        
        stats['total_captchas'] = total_original
        stats['ml_solutions_generated'] = has_ml_solution
        stats['ml_coverage_rate'] = has_ml_solution / total_original
        stats['exact_match_accuracy'] = exact_matches / total_original
        stats['conditional_accuracy'] = exact_matches / has_ml_solution if has_ml_solution > 0 else 0
        
        # Character-level accuracy
        stats['avg_char_accuracy'] = self.original_df['char_accuracy'].mean()
        stats['median_char_accuracy'] = self.original_df['char_accuracy'].median()
        
        # Error analysis
        differences_df = self.original_df[
            self.original_df['has_ml_solution'] & 
            (~self.original_df['solutions_match'])
        ].copy()
        
        if len(differences_df) > 0:
            # Add error type analysis if not already present
            if 'error_type' not in differences_df.columns:
                differences_df['error_type'] = differences_df.apply(self._classify_error, axis=1)
            error_types = differences_df['error_type'].value_counts()
            stats['error_breakdown'] = error_types.to_dict()
        else:
            stats['error_breakdown'] = {}
        
        # Length analysis
        stats['length_accuracy'] = {}
        for length in range(1, 8):
            subset = self.original_df[self.original_df['captcha_length'] == length]
            if len(subset) > 0:
                accuracy = subset['solutions_match'].mean()
                stats['length_accuracy'][f'length_{length}'] = accuracy
        
        # Character type analysis
        digit_subset = self.original_df[self.original_df['solution'].str.isdigit()]
        alpha_subset = self.original_df[self.original_df['solution'].str.isalpha()]
        mixed_subset = self.original_df[
            ~self.original_df['solution'].str.isdigit() & 
            ~self.original_df['solution'].str.isalpha()
        ]
        
        stats['digit_accuracy'] = digit_subset['solutions_match'].mean() if len(digit_subset) > 0 else 0
        stats['alpha_accuracy'] = alpha_subset['solutions_match'].mean() if len(alpha_subset) > 0 else 0
        stats['mixed_accuracy'] = mixed_subset['solutions_match'].mean() if len(mixed_subset) > 0 else 0
        
        # Character frequency analysis
        stats['character_frequency_analysis'] = self._analyze_character_frequencies()
        
        # Character type distribution
        stats['character_type_distribution'] = {
            'digits_only': len(digit_subset),
            'letters_only': len(alpha_subset), 
            'mixed': len(mixed_subset),
            'total': len(self.original_df)
        }
        
        # Save statistics
        stats_file = self.output_dir / f"accuracy_statistics_{self.timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Statistics saved to: {stats_file}")
        
        # Print summary
        print(f"\n📊 CAPTCHA Solution Accuracy Analysis")
        print(f"{'='*50}")
        print(f"Total CAPTCHAs: {stats['total_captchas']:,}")
        print(f"ML Solutions Generated: {stats['ml_solutions_generated']:,} ({stats['ml_coverage_rate']:.1%})")
        print(f"Exact Match Accuracy: {stats['exact_match_accuracy']:.1%}")
        print(f"Conditional Accuracy: {stats['conditional_accuracy']:.1%}")
        print(f"Average Character Accuracy: {stats['avg_char_accuracy']:.1%}")
        print(f"{'='*50}")
        
        # Print character frequency analysis
        char_freq = stats['character_frequency_analysis']
        char_dist = stats['character_type_distribution']
        
        print(f"\n🔤 Character Frequency Analysis")
        print(f"{'='*50}")
        print(f"Total Characters Analyzed: {char_freq['total_characters']:,}")
        print(f"")
        print(f"📊 Character Type Distribution:")
        print(f"  Digits: {char_freq['digit_stats']['count']:,} ({char_freq['digit_stats']['percentage']:.1f}%)")
        print(f"  Lowercase: {char_freq['lowercase_stats']['count']:,} ({char_freq['lowercase_stats']['percentage']:.1f}%)")
        print(f"  Uppercase: {char_freq['uppercase_stats']['count']:,} ({char_freq['uppercase_stats']['percentage']:.1f}%)")
        print(f"  Other: {char_freq['other_stats']['count']:,} ({char_freq['other_stats']['percentage']:.1f}%)")
        print(f"")
        print(f"📈 CAPTCHA Type Distribution:")
        print(f"  Digits Only: {char_dist['digits_only']:,} CAPTCHAs")
        print(f"  Letters Only: {char_dist['letters_only']:,} CAPTCHAs")
        print(f"  Mixed: {char_dist['mixed']:,} CAPTCHAs")
        print(f"")
        print(f"🎯 Accuracy by Character Type:")
        print(f"  Digits Only: {stats['digit_accuracy']:.1%}")
        print(f"  Letters Only: {stats['alpha_accuracy']:.1%}")
        print(f"  Mixed: {stats['mixed_accuracy']:.1%}")
        print(f"{'='*50}")
        
        # Show top 10 most frequent characters
        all_chars = {}
        all_chars.update(char_freq['digit_stats']['frequency_by_digit'])
        all_chars.update(char_freq['lowercase_stats']['frequency_by_letter'])
        all_chars.update(char_freq['uppercase_stats']['frequency_by_letter'])
        
        top_chars = sorted(all_chars.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        print(f"\n🏆 Top 10 Most Frequent Characters:")
        for i, (char, data) in enumerate(top_chars, 1):
            print(f"  {i:2d}. '{char}': {data['count']:,} times ({data['percentage']:.1f}%)")
        print(f"{'='*50}")
        
        return str(stats_file)
    
    def create_visualization_grids(self, max_grids: Optional[int] = None, samples_per_grid: int = 50) -> List[str]:
        """Create visualization grids for mismatched solutions"""
        logger.info("Creating visualization grids...")
        
        # Get mismatched solutions
        differences_df = self.original_df[
            self.original_df['has_ml_solution'] & 
            (~self.original_df['solutions_match'])
        ].copy()
        
        if len(differences_df) == 0:
            logger.warning("No mismatched solutions found for visualization")
            return []
        
        # Calculate total grids needed (5 columns, 10 rows = 50 samples per grid)
        cols, rows = 5, 10
        samples_per_grid = cols * rows
        total_grids_needed = (len(differences_df) + samples_per_grid - 1) // samples_per_grid
        
        # Set max_grids if not specified
        if max_grids is None:
            max_grids = total_grids_needed
        else:
            max_grids = min(max_grids, total_grids_needed)
        
        grid_files = []
        
        for grid_idx in range(max_grids):
            start_idx = grid_idx * samples_per_grid
            end_idx = min(start_idx + samples_per_grid, len(differences_df))
            
            if start_idx >= len(differences_df):
                break
                
            grid_data = differences_df.iloc[start_idx:end_idx]
            grid_file = self._create_single_grid(grid_data, grid_idx + 1, cols, rows)
            if grid_file:
                grid_files.append(grid_file)
        
        logger.info(f"Created {len(grid_files)} visualization grids")
        logger.info(f"Total mismatched solutions: {len(differences_df)}")
        return grid_files
    
    def _create_single_grid(self, data_df: pd.DataFrame, grid_num: int, cols: int, rows: int) -> Optional[str]:
        """Create a single visualization grid using matplotlib subplots with original and preprocessed images"""
        try:
            # Create figure with relaxed spacing (5 cols x 10 rows)
            # Each subplot will show both original and preprocessed images side by side
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            
            # Ensure axes is always 2D
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Process each sample
            for idx, (_, row) in enumerate(data_df.iterrows()):
                if idx >= cols * rows:
                    break
                
                grid_row = idx // cols
                grid_col = idx % cols
                ax = axes[grid_row, grid_col]
                
                self._add_captcha_comparison_to_subplot(ax, row)
            
            # Hide unused subplots
            for idx in range(len(data_df), rows * cols):
                grid_row = idx // cols
                grid_col = idx % cols
                if grid_row < rows and grid_col < cols:
                    axes[grid_row, grid_col].axis('off')
                    axes[grid_row, grid_col].set_visible(False)
            
            # Adjust spacing for better padding
            plt.subplots_adjust(hspace=0.7, wspace=0.5, top=0.95, bottom=0.05, left=0.05, right=0.95)
            
            # Save grid
            output_file = self.output_dir / f"mismatched_captchas_grid_{grid_num}_{self.timestamp}.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created grid {grid_num}: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create grid {grid_num}: {e}")
            return None
    
    def _add_captcha_comparison_to_subplot(self, ax, row):
        """Add a single CAPTCHA comparison with original and preprocessed images side by side"""
        try:
            filename = row['filename']
            original_solution = str(row['solution'])
            ml_solution = str(row['ml_solution'])
            error_type = row.get('error_type', 'unknown')
            
            # Load original image
            captcha_dir = Path("captcha_dataset")
            img_path = captcha_dir / filename
            
            # Find preprocessed image in the batch directory
            batch_dirs = [d for d in self.ml_solutions_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
            preprocessed_img = None
            
            if batch_dirs:
                latest_batch = max(batch_dirs, key=lambda x: x.stat().st_mtime)
                preprocessed_dir = latest_batch / "preprocessed"
                
                # Look for preprocessed image
                base_name = filename.split('.')[0]
                preprocessed_files = list(preprocessed_dir.glob(f"{base_name}_preprocessed.*"))
                
                if preprocessed_files:
                    try:
                        preprocessed_img = cv2.imread(str(preprocessed_files[0]), cv2.IMREAD_GRAYSCALE)
                        if preprocessed_img is not None:
                            # Resize to match original if needed
                            preprocessed_img = cv2.resize(preprocessed_img, (200, 80))
                    except Exception as e:
                        logger.debug(f"Failed to load preprocessed image: {e}")
            
            if img_path.exists():
                try:
                    # Load original image
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Create side-by-side comparison
                    if preprocessed_img is not None:
                        # Convert original to grayscale for comparison
                        if len(img_array.shape) == 3:
                            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        else:
                            img_gray = img_array
                        
                        # Resize to match if needed
                        img_gray = cv2.resize(img_gray, (200, 80))
                        
                        # Create side-by-side comparison
                        combined = np.hstack([img_gray, preprocessed_img])
                        ax.imshow(combined, cmap='gray')
                        
                        # Add dividing line
                        ax.axvline(x=200, color='red', linewidth=2, alpha=0.7)
                        
                        # Add labels
                        ax.text(100, -5, 'Original', ha='center', va='top', fontsize=8, color='blue')
                        ax.text(300, -5, 'Preprocessed', ha='center', va='top', fontsize=8, color='green')
                    else:
                        # Show only original if no preprocessed available
                        ax.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
                        ax.text(100, -5, 'Original (No Preprocessed)', ha='center', va='top', fontsize=8, color='orange')
                    
                    # Enhanced title with error type and better formatting
                    title = f"Original: {original_solution}\nML: {ml_solution}\nError: {error_type}"
                    ax.set_title(title, fontsize=10, pad=20, wrap=True)
                    ax.axis('off')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading:\n{filename}\n{str(e)[:30]}", 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=8, wrap=True)
                    ax.set_title(f"Original: {original_solution}\nML: {ml_solution}", fontsize=8)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"Image not found:\n{filename}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f"Original: {original_solution}\nML: {ml_solution}", fontsize=8)
                ax.axis('off')
                
        except Exception as e:
            # Fallback for any other errors
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')

    def update_consolidated_csv(self, consolidated_csv_path="captcha_solutions_consolidated.csv"):
        """Update the existing consolidated CSV with ML solution results."""
        logger.info(f"Updating consolidated CSV: {consolidated_csv_path}")
        
        # Read the existing consolidated CSV
        try:
            consolidated_df = pd.read_csv(consolidated_csv_path)
            logger.info(f"Loaded {len(consolidated_df)} records from consolidated CSV")
        except FileNotFoundError:
            logger.error(f"Consolidated CSV not found: {consolidated_csv_path}")
            return None
        
        # Create a copy for updating
        updated_df = consolidated_df.copy()
        
        # Add ML solution columns if they don't exist
        if 'ml_solution' not in updated_df.columns:
            updated_df['ml_solution'] = None
        if 'ml_success' not in updated_df.columns:
            updated_df['ml_success'] = False
        if 'ml_accuracy' not in updated_df.columns:
            updated_df['ml_accuracy'] = None
        if 'ml_match' not in updated_df.columns:
            updated_df['ml_match'] = False
        if 'ml_solve_time' not in updated_df.columns:
            updated_df['ml_solve_time'] = None
        if 'ml_error_type' not in updated_df.columns:
            updated_df['ml_error_type'] = None
        
        # Update with ML results from the ml_solutions dictionary
        matches_found = 0
        for idx, row in updated_df.iterrows():
            filename = row['filename']
            if filename in self.ml_solutions:
                ml_solution = self.ml_solutions[filename]
                updated_df.at[idx, 'ml_solution'] = ml_solution
                updated_df.at[idx, 'ml_success'] = True
                
                # Calculate accuracy and match
                original_solution = str(row['solution']).lower()
                ml_solution_lower = str(ml_solution).lower()
                updated_df.at[idx, 'ml_match'] = (original_solution == ml_solution_lower)
                updated_df.at[idx, 'ml_accuracy'] = self._calculate_char_accuracy(original_solution, ml_solution_lower)
                
                # Add error type for mismatches
                if not updated_df.at[idx, 'ml_match']:
                    error_row = pd.Series({
                        'solution': original_solution,
                        'ml_solution': ml_solution_lower
                    })
                    updated_df.at[idx, 'ml_error_type'] = self._classify_error(error_row)
                
                matches_found += 1
        
        # Save updated consolidated CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"captcha_solutions_consolidated_with_ml_{timestamp}.csv"
        updated_df.to_csv(output_path, index=False)
        
        logger.info(f"Updated consolidated CSV saved to: {output_path}")
        logger.info(f"Found ML solutions for {matches_found}/{len(updated_df)} records")
        
        # Generate summary statistics
        ml_coverage = (updated_df['ml_success'].sum() / len(updated_df)) * 100
        ml_accuracy = updated_df[updated_df['ml_success']]['ml_match'].mean() * 100
        
        print(f"\n📊 Updated Consolidated CSV Summary")
        print(f"==================================================")
        print(f"Total Records: {len(updated_df):,}")
        print(f"ML Solutions: {updated_df['ml_success'].sum():,} ({ml_coverage:.1f}%)")
        print(f"ML Accuracy: {ml_accuracy:.1f}% (exact matches)")
        print(f"Output: {output_path}")
        print(f"==================================================")
        
        return output_path


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Compare ML solutions with original solutions")
    parser.add_argument("--original-csv", default="captcha_solutions_consolidated.csv", 
                       help="Original consolidated CSV file")
    parser.add_argument("--ml-solutions-dir", default="output/captcha_solutions", 
                       help="Directory containing ML solution results")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--max-grids", type=int, default=None, help="Maximum number of visualization grids (default: all needed)")
    parser.add_argument("--no-grids", action="store_true", help="Skip visualization grid creation")
    parser.add_argument("--update-consolidated", action="store_true", help="Update the consolidated CSV with ML results")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SolutionComparisonAnalyzer(
        original_csv=args.original_csv,
        ml_solutions_dir=args.ml_solutions_dir,
        output_dir=args.output_dir
    )
    
    try:
        # Create merged comparison CSV
        merged_csv = analyzer.create_merged_comparison_csv()
        print(f"✓ Merged comparison CSV: {merged_csv}")
        
        # Create differences CSV
        differences_csv = analyzer.create_differences_csv()
        print(f"✓ Differences CSV: {differences_csv}")
        
        # Generate statistics
        stats_file = analyzer.generate_statistics_report()
        print(f"✓ Statistics report: {stats_file}")
        
        # Update consolidated CSV if requested
        if args.update_consolidated:
            consolidated_csv = analyzer.update_consolidated_csv(args.original_csv)
            if consolidated_csv:
                print(f"✓ Updated consolidated CSV: {consolidated_csv}")
        
        # Create visualization grids
        if not args.no_grids:
            grid_files = analyzer.create_visualization_grids(max_grids=args.max_grids)
            if grid_files:
                print(f"✓ Created {len(grid_files)} visualization grids")
                for grid_file in grid_files:
                    print(f"  - {grid_file}")
            else:
                print("⚠ No visualization grids created")
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"All outputs saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()