"""
Performance benchmarking for MGDStars package.

This module provides benchmarking tools to monitor performance
across different models and parameter ranges.
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from mgdstars import MGDStars


class MGDStarsBenchmark:
    """Benchmark suite for MGDStars performance analysis."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_models(self, 
                        test_configs: List[Dict],
                        models: List[str] = None,
                        alpha_values: List[float] = None) -> Dict:
        """
        Benchmark all models across test configurations.
        
        Parameters:
        -----------
        test_configs : list
            List of test configurations with M, R values
        models : list, optional
            Models to test (default: all three)
        alpha_values : list, optional
            Alpha values to test for each model
            
        Returns:
        --------
        dict : Comprehensive benchmark results
        """
        if models is None:
            models = ['TolmanIV', 'HeintzmannIIa', 'WymanIIa']
        
        if alpha_values is None:
            alpha_values = [0.1, 0.01, 0.005]  # Model-appropriate defaults
        
        results = {}
        
        for model in models:
            print(f"Benchmarking {model}...")
            results[model] = self._benchmark_single_model(
                model, test_configs, alpha_values
            )
        
        self.results = results
        return results
    
    def _benchmark_single_model(self, 
                               model: str, 
                               test_configs: List[Dict],
                               alpha_values: List[float]) -> Dict:
        """Benchmark a single model."""
        model_results = {
            'success_count': 0,
            'total_attempts': 0,
            'solve_times': [],
            'residuals': [],
            'failed_configs': [],
            'alpha_performance': {}
        }
        
        for alpha in alpha_values:
            alpha_results = {
                'success_count': 0,
                'total_attempts': 0,
                'solve_times': [],
                'residuals': []
            }
            
            for config in test_configs:
                try:
                    star = MGDStars(alpha=alpha, seed_model=model)
                    
                    start_time = time.time()
                    result = star.solve_star_configuration(
                        M_val=config['M'], 
                        R_val=config['R'],
                        verbose=False
                    )
                    solve_time = time.time() - start_time
                    
                    alpha_results['total_attempts'] += 1
                    model_results['total_attempts'] += 1
                    
                    if result['success']:
                        alpha_results['success_count'] += 1
                        model_results['success_count'] += 1
                        alpha_results['solve_times'].append(solve_time)
                        alpha_results['residuals'].append(result['residual'])
                        model_results['solve_times'].append(solve_time)
                        model_results['residuals'].append(result['residual'])
                    else:
                        model_results['failed_configs'].append({
                            'config': config,
                            'alpha': alpha,
                            'error': result.get('error', 'Unknown')
                        })
                        
                except Exception as e:
                    model_results['failed_configs'].append({
                        'config': config,
                        'alpha': alpha,
                        'error': str(e)
                    })
            
            model_results['alpha_performance'][alpha] = alpha_results
        
        return model_results
    
    def get_performance_summary(self) -> Dict:
        """Generate performance summary across all models."""
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmark_models() first.")
        
        summary = {}
        
        for model, data in self.results.items():
            if data['total_attempts'] > 0:
                success_rate = data['success_count'] / data['total_attempts'] * 100
                avg_time = np.mean(data['solve_times']) if data['solve_times'] else np.inf
                avg_residual = np.mean(data['residuals']) if data['residuals'] else np.inf
                
                summary[model] = {
                    'success_rate': success_rate,
                    'avg_solve_time': avg_time,
                    'avg_residual': avg_residual,
                    'total_attempts': data['total_attempts'],
                    'successful_solves': data['success_count']
                }
        
        return summary
    
    def print_performance_report(self):
        """Print detailed performance report."""
        summary = self.get_performance_summary()
        
        print("="*70)
        print("MGD STARS PERFORMANCE BENCHMARK REPORT")
        print("="*70)
        
        print(f"{'Model':<15} {'Success Rate':<12} {'Avg Time (s)':<12} {'Avg Residual':<13}")
        print("-" * 60)
        
        for model, stats in summary.items():
            print(f"{model:<15} {stats['success_rate']:<11.1f}% "
                  f"{stats['avg_solve_time']:<11.4f} {stats['avg_residual']:<12.2e}")
        
        print("\nDetailed Analysis:")
        print("-" * 40)
        
        for model, data in self.results.items():
            print(f"\n{model}:")
            print(f"  Total configurations tested: {data['total_attempts']}")
            print(f"  Successful solutions: {data['success_count']}")
            print(f"  Failed attempts: {len(data['failed_configs'])}")
            
            if data['solve_times']:
                print(f"  Time range: {min(data['solve_times']):.4f}s - {max(data['solve_times']):.4f}s")
            
            if data['failed_configs']:
                print(f"  Common failure reasons:")
                error_counts = {}
                for failure in data['failed_configs']:
                    error = failure['error'][:50]  # Truncate long errors
                    error_counts[error] = error_counts.get(error, 0) + 1
                
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    - {error}: {count} occurrences")


def quick_benchmark(M_val=1.4, R_val=12.0, alpha_range=None):
    """
    Quick benchmark for a single configuration.
    
    Parameters:
    -----------
    M_val, R_val : float
        Mass and radius values
    alpha_range : list, optional
        Alpha values to test
        
    Returns:
    --------
    dict : Benchmark results
    """
    if alpha_range is None:
        alpha_range = [0.1, 0.05, 0.01, 0.005]
    
    models = ['TolmanIV', 'HeintzmannIIa', 'WymanIIa']
    results = {}
    
    print(f"Quick benchmark for M={M_val}, R={R_val}")
    print("="*50)
    
    for model in models:
        print(f"\n{model}:")
        model_results = []
        
        for alpha in alpha_range:
            try:
                star = MGDStars(alpha=alpha, seed_model=model)
                start_time = time.time()
                result = star.solve_star_configuration(M_val, R_val, verbose=False)
                solve_time = time.time() - start_time
                
                if result['success']:
                    model_results.append({
                        'alpha': alpha,
                        'success': True,
                        'time': solve_time,
                        'residual': result['residual']
                    })
                    print(f"  α={alpha:6.3f}: ✓ ({solve_time:.3f}s, residual={result['residual']:.2e})")
                else:
                    model_results.append({
                        'alpha': alpha,
                        'success': False,
                        'time': solve_time,
                        'error': result.get('error', 'Unknown')
                    })
                    print(f"  α={alpha:6.3f}: ✗ ({solve_time:.3f}s)")
                    
            except Exception as e:
                print(f"  α={alpha:6.3f}: ✗ Exception: {str(e)[:50]}")
        
        results[model] = model_results
    
    return results


if __name__ == '__main__':
    # Example usage
    benchmark = MGDStarsBenchmark()
    
    # Test configurations
    test_configs = [
        {"M": 1.4, "R": 12.0},
        {"M": 1.0, "R": 10.0},
        {"M": 2.0, "R": 11.0},
        {"M": 1.6, "R": 13.0},
    ]
    
    # Run benchmark
    results = benchmark.benchmark_models(test_configs)
    benchmark.print_performance_report()
