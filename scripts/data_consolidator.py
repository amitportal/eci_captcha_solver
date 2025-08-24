#!/usr/bin/env python3
"""
scripts/data_consolidator.py - CAPTCHA Dataset Consolidation Tool
Creates consolidated CSV from individual PNG/JSON files for training
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional
from datetime import datetime
import cv2
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def process_file_pair_async(json_file: Path, dataset_dir: Path, consolidator) -> Optional[Dict]:
    """Process a single JSON/PNG file pair asynchronously"""
    try:
        # Load JSON solution
        with open(json_file, 'r', encoding='utf-8') as f:
            solution_data = json.load(f)
        
        # Get corresponding PNG file
        base_name = json_file.stem
        png_file = dataset_dir / f"{base_name}.png"
        
        if not png_file.exists():
            logger.warning(f"Missing PNG file: {png_file}")
            return None
        
        # Validate solution
        solution = solution_data.get('solution', '').strip()
        if not consolidator.validate_solution(solution):
            return None
        
        # Validate image
        image_metadata = consolidator.validate_image_file(png_file)
        if not image_metadata:
            return None
        
        # Create consolidated entry
        entry = {
            # File information
            "filename": png_file.name,
            "json_file": json_file.name,
            "source_json": json_file.stem,
            
            # Solution information
            "solution": solution.lower(),  # Normalize to lowercase
            "original_solution": solution,  # Keep original case
            "captcha_length": len(solution),
            
            # Character analysis
            "digits": ''.join(c for c in solution if c.isdigit()),
            "digits_length": len([c for c in solution if c.isdigit()]),
            "characters": ''.join(c for c in solution if c.isalpha()).lower(),
            "char_length": len([c for c in solution if c.isalpha()]),
            
            # Image information
            "image_width": image_metadata["width"],
            "image_height": image_metadata["height"],
            "image_size": image_metadata["file_size"],
            
            # Original metadata
            "task_id": solution_data.get('task_id'),
            "instance_id": solution_data.get('instance_id'),
            "timestamp": solution_data.get('timestamp'),
            "state_code": solution_data.get('state_code'),
            "source": solution_data.get('source', 'unknown')
        }
        
        return entry
        
    except Exception as e:
        logger.error(f"Error processing {json_file}: {e}")
        return None

class CaptchaDataConsolidator:
    """Consolidate CAPTCHA training data from individual files"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        logger.info(f"Consolidating data from: {self.dataset_dir}")
    
    def validate_image_file(self, image_path: Path) -> Optional[Dict]:
        """Validate PNG file and extract metadata"""
        try:
            # Load image to check validity
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            height, width = img.shape[:2]
            file_size = image_path.stat().st_size
            
            # Check if dimensions are reasonable (around 200x80)
            if width < 100 or height < 40:
                logger.warning(f"Suspicious image dimensions {width}x{height}: {image_path.name}")
            
            return {
                "width": width,
                "height": height,
                "file_size": file_size,
                "is_valid": True
            }
            
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return None
    
    def validate_solution(self, solution: str) -> bool:
        """Validate CAPTCHA solution"""
        if not solution:
            return False
        
        # Remove whitespace
        solution = solution.strip()
        
        # Check length (should be exactly 6)
        if len(solution) != 6:
            return False
        
        # Check if alphanumeric
        if not solution.replace(' ', '').isalnum():
            return False
        
        return True
    
    def scan_dataset_files(self) -> List[Dict]:
        """Scan dataset directory and validate files with parallel processing"""
        logger.info("Scanning dataset files with parallel processing...")
        
        # Get all JSON files
        json_files = list(self.dataset_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        if not json_files:
            logger.warning("No JSON files found in dataset directory")
            return []
        
        consolidated_data = []
        processed_count = 0
        valid_count = 0
        invalid_solutions = 0
        missing_images = 0
        corrupted_images = 0
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(32, len(json_files))  # Limit concurrent workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file_pair_async, json_file, self.dataset_dir, self): json_file 
                for json_file in json_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(json_files)} files...")
                
                try:
                    result = future.result()
                    if result:
                        consolidated_data.append(result)
                        valid_count += 1
                    else:
                        # Check what type of failure
                        json_file = future_to_file[future]
                        png_file = json_file.with_suffix('.png')
                        if not png_file.exists():
                            missing_images += 1
                        else:
                            invalid_solutions += 1
                            
                except Exception as e:
                    logger.error(f"Error processing file {future_to_file[future]}: {e}")
                    corrupted_images += 1
        
        # Log statistics
        logger.info(f"\nDataset consolidation complete:")
        logger.info(f"  Total files processed: {processed_count}")
        logger.info(f"  Valid entries: {valid_count}")
        logger.info(f"  Invalid solutions: {invalid_solutions}")
        logger.info(f"  Missing images: {missing_images}")
        logger.info(f"  Corrupted images: {corrupted_images}")
        
        return consolidated_data

    def analyze_dataset_quality(self, data: List[Dict]) -> Dict:
        """Analyze dataset quality and characteristics"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        analysis = {
            "total_samples": int(len(df)),
            "unique_solutions": int(df['solution'].nunique()),
            "duplicate_solutions": int(len(df) - df['solution'].nunique()),
            
            # Image dimensions analysis
            "image_dimensions": {
                "width_stats": {
                    "mean": float(df['image_width'].mean()),
                    "min": int(df['image_width'].min()),
                    "max": int(df['image_width'].max()),
                    "std": float(df['image_width'].std())
                },
                "height_stats": {
                    "mean": float(df['image_height'].mean()),
                    "min": int(df['image_height'].min()),
                    "max": int(df['image_height'].max()),
                    "std": float(df['image_height'].std())
                },
                "common_dimensions": {f"{k[0]}x{k[1]}": int(v) for k, v in df.groupby(['image_width', 'image_height']).size().items() if isinstance(k, tuple)}
            },
            
            # Character distribution
            "character_distribution": {
                "avg_digits_per_captcha": float(df['digits_length'].mean()),
                "avg_letters_per_captcha": float(df['char_length'].mean()),
                "digits_distribution": {str(k): int(v) for k, v in df['digits_length'].value_counts().items()},
                "letters_distribution": {str(k): int(v) for k, v in df['char_length'].value_counts().items()}
            },
            
            # Case distribution
            "case_analysis": {
                "all_lowercase": int(len(df[df['solution'].str.islower()])),
                "all_uppercase": int(len(df[df['solution'].str.isupper()])),
                "mixed_case": int(len(df[~df['solution'].str.islower() & ~df['solution'].str.isupper()])),
                "contains_digits": int(len(df[df['digits_length'] > 0]))
            },
            
            # Source distribution
            "source_distribution": {str(k): int(v) for k, v in df['source'].value_counts().items()},
            
            # State distribution (if available)
            "state_distribution": {str(k): int(v) for k, v in df['state_code'].value_counts().items()} if 'state_code' in df.columns else {},
            
            # File size statistics
            "file_size_stats": {
                "mean_kb": float(df['image_size'].mean() / 1024),
                "min_kb": float(df['image_size'].min() / 1024),
                "max_kb": float(df['image_size'].max() / 1024),
                "std_kb": float(df['image_size'].std() / 1024)
            }
        }
        
        # Character frequency analysis
        all_chars = ''.join(df['solution'])
        char_freq = {}
        for char in set(all_chars):
            char_freq[char] = all_chars.count(char)
        
        analysis["character_frequency"] = dict(sorted(char_freq.items(), key=lambda x: x[1], reverse=True))
        
        return analysis
    
    def create_consolidated_files(self, output_dir: Optional[str] = None):
        """Create consolidated JSON and CSV files"""
        if output_dir is None:
            output_dir_path = self.dataset_dir.parent
        else:
            output_dir_path = Path(output_dir)
        
        output_dir_path.mkdir(exist_ok=True)
        
        # Scan and consolidate data
        logger.info("Starting dataset consolidation...")
        consolidated_data = self.scan_dataset_files()
        
        if not consolidated_data:
            logger.error("No valid data found to consolidate!")
            return None, None, None
        
        # Analyze dataset quality
        logger.info("Analyzing dataset quality...")
        quality_analysis = self.analyze_dataset_quality(consolidated_data)
        
        # Save consolidated JSON
        json_output = output_dir_path / "captcha_solutions_consolidated.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved consolidated JSON: {json_output}")
        
        # Save consolidated CSV
        csv_output = output_dir_path / "captcha_solutions_consolidated.csv"
        df = pd.DataFrame(consolidated_data)
        df.to_csv(csv_output, index=False, encoding='utf-8')
        
        logger.info(f"✓ Saved consolidated CSV: {csv_output}")
        
        # Save quality analysis
        analysis_output = output_dir_path / "dataset_quality_analysis.json"
        quality_analysis["generation_timestamp"] = datetime.now().isoformat()
        with open(analysis_output, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(quality_analysis), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved quality analysis: {analysis_output}")
        
        # Print summary
        self.print_dataset_summary(quality_analysis)
        
        return json_output, csv_output, analysis_output
    
    def print_dataset_summary(self, analysis: Dict):
        """Print dataset summary"""
        print("\n" + "="*60)
        print("📊 DATASET CONSOLIDATION SUMMARY")
        print("="*60)
        
        print(f"\n📋 Basic Statistics:")
        print(f"  Total samples: {analysis['total_samples']:,}")
        print(f"  Unique solutions: {analysis['unique_solutions']:,}")
        print(f"  Duplicate solutions: {analysis['duplicate_solutions']:,}")
        
        print(f"\n🖼️  Image Dimensions:")
        width_stats = analysis['image_dimensions']['width_stats']
        height_stats = analysis['image_dimensions']['height_stats']
        print(f"  Width: {width_stats['mean']:.1f}±{width_stats['std']:.1f} (range: {width_stats['min']}-{width_stats['max']})")
        print(f"  Height: {height_stats['mean']:.1f}±{height_stats['std']:.1f} (range: {height_stats['min']}-{height_stats['max']})")
        
        # Most common dimensions
        common_dims = analysis['image_dimensions']['common_dimensions']
        most_common = max(common_dims.items(), key=lambda x: x[1])
        print(f"  Most common: {most_common[0][0]}x{most_common[0][1]} ({most_common[1]:,} images)")
        
        print(f"\n🔤 Character Analysis:")
        char_dist = analysis['character_distribution']
        print(f"  Avg digits per CAPTCHA: {char_dist['avg_digits_per_captcha']:.1f}")
        print(f"  Avg letters per CAPTCHA: {char_dist['avg_letters_per_captcha']:.1f}")
        
        case_analysis = analysis['case_analysis']
        total = analysis['total_samples']
        print(f"\n📝 Case Distribution:")
        print(f"  All lowercase: {case_analysis['all_lowercase']:,} ({case_analysis['all_lowercase']/total*100:.1f}%)")
        print(f"  All uppercase: {case_analysis['all_uppercase']:,} ({case_analysis['all_uppercase']/total*100:.1f}%)")
        print(f"  Mixed case: {case_analysis['mixed_case']:,} ({case_analysis['mixed_case']/total*100:.1f}%)")
        print(f"  Contains digits: {case_analysis['contains_digits']:,} ({case_analysis['contains_digits']/total*100:.1f}%)")
        
        print(f"\n📁 File Size:")
        file_stats = analysis['file_size_stats']
        print(f"  Average: {file_stats['mean_kb']:.1f} KB")
        print(f"  Range: {file_stats['min_kb']:.1f} - {file_stats['max_kb']:.1f} KB")
        
        print(f"\n🔥 Top 10 Characters:")
        char_freq = analysis['character_frequency']
        total_chars = sum(char_freq.values())
        for i, (char, count) in enumerate(list(char_freq.items())[:10]):
            print(f"  '{char}': {count:,} ({count/total_chars*100:.1f}%)")
        
        print("\n✅ Dataset ready for training!")
        print("="*60)

def main():
    """Main consolidation function"""
    parser = argparse.ArgumentParser(description="CAPTCHA Dataset Consolidation Tool")
    parser.add_argument("dataset_dir", help="Directory containing PNG and JSON files")
    parser.add_argument("--output-dir", help="Output directory (default: parent of dataset dir)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate files without creating outputs")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum required samples")
    
    args = parser.parse_args()
    
    try:
        consolidator = CaptchaDataConsolidator(args.dataset_dir)
        
        if args.validate_only:
            # Only validate files
            data = consolidator.scan_dataset_files()
            analysis = consolidator.analyze_dataset_quality(data)
            
            if len(data) >= args.min_samples:
                print(f"✅ Validation passed: {len(data)} valid samples found")
                consolidator.print_dataset_summary(analysis)
                return 0
            else:
                print(f"❌ Validation failed: Only {len(data)} samples found (minimum: {args.min_samples})")
                return 1
        else:
            # Create consolidated files
            json_file, csv_file, analysis_file = consolidator.create_consolidated_files(args.output_dir)
            
            if json_file and csv_file:
                print(f"\n🎉 Consolidation completed successfully!")
                print(f"📄 Files created:")
                print(f"  JSON: {json_file}")
                print(f"  CSV: {csv_file}")
                print(f"  Analysis: {analysis_file}")
                print(f"\n🚀 Next step: python scripts/batch_trainer.py")
                return 0
            else:
                print("❌ Consolidation failed!")
                return 1
                
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())