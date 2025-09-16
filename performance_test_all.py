#!/usr/bin/env python3
"""
Performance test script for ECI CAPTCHA solver
Tests all CAPTCHAs in the consolidated CSV with all permutations of 
preprocessing methods and solution methods
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import itertools

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ProductionCaptchaSolver, OptimizedCaptchaConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define available methods
PREPROCESSING_METHODS = ["grayscale", "light_smoothing", "precise_mathematical", "raw"]
SOLUTION_METHODS = ["segment", "crnn", "ocr"]

class PerformanceTester:
    """Test CAPTCHA solver performance across all images in the dataset"""
    
    def __init__(self, dataset_dir: str = "captcha_dataset", 
                 csv_file: str = "captcha_solutions_consolidated.csv",
                 model_dir: str = "models"):
        self.dataset_dir = Path(dataset_dir)
        self.csv_file = Path(csv_file)
        self.model_dir = Path(model_dir)
        self.results = []
        
        # Load the consolidated CSV
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} CAPTCHA entries from {self.csv_file}")
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    def _calculate_accuracy(self, original: str, predicted: str) -> float:
        """Calculate character-level accuracy between original and predicted solutions"""
        if not original or not predicted:
            return 0.0
            
        original = str(original).lower()
        predicted = str(predicted).lower()
        
        # Exact match
        if original == predicted:
            return 1.0
            
        # Character-level accuracy
        max_len = max(len(original), len(predicted))
        correct_chars = sum(1 for o, p in zip(original, predicted) if o == p)
        return correct_chars / max_len if max_len > 0 else 0.0
    
    def _solve_with_method(self, solver, img_path, confidence_threshold, solution_method):
        """Solve CAPTCHA with a specific solution method"""
        method_used = None
        solution = None
        if solution_method == "segment":
            solution = solver._solve_with_segmentation(str(img_path), confidence_threshold)
            method_used = "segment"
        elif solution_method == "crnn":
            solution = solver._solve_with_crnn(str(img_path), confidence_threshold)
            method_used = "crnn"
        elif solution_method == "ocr":
            solution = solver._solve_with_ocr_fallback(str(img_path))
            method_used = "ocr"
        else:
            # Default fallback sequence
            if solver.segmentation_model:
                solution = solver._solve_with_segmentation(str(img_path), confidence_threshold)
                method_used = "segment" if solution else None
            if not solution and solver.crnn_model:
                solution = solver._solve_with_crnn(str(img_path), confidence_threshold)
                method_used = "crnn" if solution else method_used
            if not solution and solver.fallback_api_key:
                solution = solver._solve_with_api(str(img_path))
                method_used = "api" if solution else method_used
            if not solution:
                solution = solver._solve_with_ocr_fallback(str(img_path))
                method_used = "ocr" if solution else method_used
        return solution, method_used
    
    def test_all_captchas(self, num_samples: Optional[int] = None, 
                         preprocessing_methods: List[str] = None,
                         solution_methods: List[str] = None) -> Dict:
        """Test all CAPTCHAs in the dataset with all permutations of preprocessing and solution methods"""
        
        if preprocessing_methods is None:
            preprocessing_methods = PREPROCESSING_METHODS
            
        if solution_methods is None:
            solution_methods = SOLUTION_METHODS
            
        # Create all permutations of preprocessing and solution methods
        method_permutations = list(itertools.product(preprocessing_methods, solution_methods))
        logger.info(f"Testing {len(method_permutations)} method permutations")
        
        # Limit samples if specified
        test_df = self.df.sample(n=num_samples, random_state=42) if num_samples else self.df
        logger.info(f"Testing {len(test_df)} CAPTCHAs")
        
        # Initialize solvers for each preprocessing method
        solvers = {}
        for method in preprocessing_methods:
            try:
                solvers[method] = ProductionCaptchaSolver(
                    model_dir=str(self.model_dir),
                    preprocessing_method=method
                )
                logger.info(f"✓ Initialized {method} solver")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {method} solver: {e}")
                solvers[method] = None
        
        # Test each CAPTCHA
        results = []
        for idx, row in test_df.iterrows():
            filename = row['filename']
            original_solution = str(row['solution']).strip().lower()
            
            # Skip if solution is not 6 characters
            if len(original_solution) != OptimizedCaptchaConfig.CAPTCHA_LENGTH:
                logger.debug(f"Skipping {filename}: solution length is not {OptimizedCaptchaConfig.CAPTCHA_LENGTH}")
                continue
                
            img_path = self.dataset_dir / filename
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            result_entry = {
                'filename': filename,
                'original_solution': original_solution,
                'methods': {}
            }
            
            # Test with each method permutation
            for preprocessing_method, solution_method in method_permutations:
                # Use the solver for the preprocessing method
                solver = solvers[preprocessing_method]
                
                if solver is None:
                    method_key = f"{preprocessing_method}_{solution_method}"
                    result_entry['methods'][method_key] = {
                        'solution': None,
                        'success': False,
                        'accuracy': 0.0,
                        'time': 0.0,
                        'error': 'Solver initialization failed'
                    }
                    continue
                    
                try:
                    start_time = time.time()
                    solution, method_used = self._solve_with_method(
                        solver, img_path, 
                        OptimizedCaptchaConfig.CONFIDENCE_THRESHOLD, 
                        solution_method
                    )
                    solve_time = time.time() - start_time
                    
                    method_key = f"{preprocessing_method}_{solution_method}"
                    
                    if solution:
                        accuracy = self._calculate_accuracy(original_solution, solution)
                        result_entry['methods'][method_key] = {
                            'solution': solution,
                            'success': True,
                            'accuracy': accuracy,
                            'time': solve_time,
                            'match': solution.lower() == original_solution
                        }
                    else:
                        result_entry['methods'][method_key] = {
                            'solution': None,
                            'success': False,
                            'accuracy': 0.0,
                            'time': solve_time,
                            'error': 'No solution found'
                        }
                        
                except Exception as e:
                    logger.error(f"Error solving {filename} with {preprocessing_method}_{solution_method}: {e}")
                    method_key = f"{preprocessing_method}_{solution_method}"
                    result_entry['methods'][method_key] = {
                        'solution': None,
                        'success': False,
                        'accuracy': 0.0,
                        'time': 0.0,
                        'error': str(e)
                    }
            
            results.append(result_entry)
            
            # Log progress
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_df)} CAPTCHAs")
        
        self.results = results
        return self._generate_summary(results)
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from test results"""
        summary = {
            'total_captchas': len(results),
            'methods': {}
        }
        
        # Initialize method statistics
        methods = set()
        for result in results:
            methods.update(result['methods'].keys())
            
        for method in methods:
            summary['methods'][method] = {
                'successful_solves': 0,
                'total_accuracy': 0.0,
                'total_time': 0.0,
                'exact_matches': 0,
                'failed_solves': 0
            }
        
        # Aggregate statistics
        for result in results:
            for method, method_result in result['methods'].items():
                if method_result['success']:
                    summary['methods'][method]['successful_solves'] += 1
                    summary['methods'][method]['total_accuracy'] += method_result['accuracy']
                    summary['methods'][method]['total_time'] += method_result['time']
                    if method_result.get('match', False):
                        summary['methods'][method]['exact_matches'] += 1
                else:
                    summary['methods'][method]['failed_solves'] += 1
        
        # Calculate averages
        for method, stats in summary['methods'].items():
            if stats['successful_solves'] > 0:
                stats['avg_accuracy'] = stats['total_accuracy'] / stats['successful_solves']
                stats['avg_time'] = stats['total_time'] / stats['successful_solves']
                stats['success_rate'] = stats['successful_solves'] / summary['total_captchas']
                stats['exact_match_rate'] = stats['exact_matches'] / summary['total_captchas']
            else:
                stats['avg_accuracy'] = 0.0
                stats['avg_time'] = 0.0
                stats['success_rate'] = 0.0
                stats['exact_match_rate'] = 0.0
        
        return summary
    
    def save_results(self, output_dir: str = "performance_results") -> str:
        """Save test results and summary to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate and save summary
        summary = self._generate_summary(self.results)
        summary_file = output_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as CSV for easier analysis
        csv_data = []
        for result in self.results:
            row = {
                'filename': result['filename'],
                'original_solution': result['original_solution']
            }
            
            for method, method_result in result['methods'].items():
                prefix = f"{method}_"
                row[prefix + 'solution'] = method_result.get('solution', '')
                row[prefix + 'success'] = method_result.get('success', False)
                row[prefix + 'accuracy'] = method_result.get('accuracy', 0.0)
                row[prefix + 'time'] = method_result.get('time', 0.0)
                row[prefix + 'match'] = method_result.get('match', False)
                row[prefix + 'error'] = method_result.get('error', '')
            
            csv_data.append(row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = output_path / f"performance_results_{timestamp}.csv"
        csv_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {output_path}")
        return str(output_path)
    
    def print_summary(self, summary: Dict):
        """Print a formatted summary of the test results"""
        print("\n" + "="*80)
        print("ECI CAPTCHA SOLVER PERFORMANCE TEST SUMMARY")
        print("="*80)
        print(f"Total CAPTCHAs tested: {summary['total_captchas']}")
        print("-"*80)
        
        # Sort methods by success rate for better readability
        sorted_methods = sorted(
            summary['methods'].items(), 
            key=lambda x: x[1]['success_rate'], 
            reverse=True
        )
        
        for method, stats in sorted_methods:
            print(f"\n{method.upper()} METHOD:")
            print(f"  Successful solves: {stats['successful_solves']}/{summary['total_captchas']} "
                  f"({stats['success_rate']*100:.1f}%)")
            print(f"  Exact matches: {stats['exact_matches']}/{summary['total_captchas']} "
                  f"({stats['exact_match_rate']*100:.1f}%)")
            print(f"  Average accuracy: {stats['avg_accuracy']*100:.1f}%")
            print(f"  Average solve time: {stats['avg_time']*1000:.1f}ms")
            print(f"  Failed solves: {stats['failed_solves']}")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Performance test for ECI CAPTCHA solver")
    parser.add_argument("--dataset-dir", default="captcha_dataset", 
                       help="CAPTCHA dataset directory")
    parser.add_argument("--csv-file", default="captcha_solutions_consolidated.csv",
                       help="Consolidated CSV file with CAPTCHA solutions")
    parser.add_argument("--model-dir", default="models",
                       help="Model directory")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to test (default: all)")
    parser.add_argument("--preprocessing-methods", nargs="+", 
                       default=PREPROCESSING_METHODS,
                       help="Preprocessing methods to test")
    parser.add_argument("--solution-methods", nargs="+", 
                       default=SOLUTION_METHODS,
                       help="Solution methods to test")
    parser.add_argument("--output-dir", default="performance_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Create tester
        tester = PerformanceTester(
            dataset_dir=args.dataset_dir,
            csv_file=args.csv_file,
            model_dir=args.model_dir
        )
        
        # Run tests
        logger.info("Starting performance tests...")
        summary = tester.test_all_captchas(
            num_samples=args.num_samples,
            preprocessing_methods=args.preprocessing_methods,
            solution_methods=args.solution_methods
        )
        
        # Save results
        output_path = tester.save_results(args.output_dir)
        
        # Print summary
        tester.print_summary(summary)
        
        print(f"\nDetailed results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise

if __name__ == "__main__":
    main()