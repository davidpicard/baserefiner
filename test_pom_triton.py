"""
Test and benchmark suite for Triton-optimized PoM implementation.

Tests correctness against the original implementation and measures
speedup with statistical significance testing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple
from scipy import stats
import json
from pathlib import Path

# Import original PoM functions
from pom import pom_activation, po2, po3, po4, polynomial_aggregation_, polynomial_selection_

# Will be imported after creation
pom_triton = None


class BenchmarkResults:
    """Store and analyze benchmark results."""
    
    def __init__(self):
        self.results: Dict[str, List[float]] = {}  # key: (seq_len, dim), value: list of times
        self.triton_results: Dict[str, List[float]] = {}
        
    def add_result(self, seq_len: int, dim: int, time_ms: float, is_triton: bool = False):
        """Add a benchmark result."""
        key = f"{seq_len}_{dim}"
        if is_triton:
            if key not in self.triton_results:
                self.triton_results[key] = []
            self.triton_results[key].append(time_ms)
        else:
            if key not in self.results:
                self.results[key] = []
            self.results[key].append(time_ms)
    
    def get_stats(self, key: str, is_triton: bool = False) -> Dict:
        """Get statistics for a configuration."""
        data = self.triton_results[key] if is_triton else self.results[key]
        data = np.array(data)
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'count': len(data),
        }
    
    def compute_speedup(self, seq_len: int, dim: int) -> Dict:
        """Compute speedup and statistical significance."""
        key = f"{seq_len}_{dim}"
        if key not in self.results or key not in self.triton_results:
            return None
        
        orig_times = np.array(self.results[key])
        triton_times = np.array(self.triton_results[key])
        
        speedup = orig_times.mean() / triton_times.mean()
        
        # Perform t-test for statistical significance
        t_stat, p_value = stats.ttest_ind(orig_times, triton_times)
        
        # Calculate confidence interval for speedup
        # Using bootstrapping for speedup CI
        n_bootstrap = 1000
        speedups = []
        for _ in range(n_bootstrap):
            orig_sample = np.random.choice(orig_times, size=len(orig_times), replace=True)
            triton_sample = np.random.choice(triton_times, size=len(triton_times), replace=True)
            speedups.append(orig_sample.mean() / triton_sample.mean())
        
        speedups = np.array(speedups)
        speedup_ci = (np.percentile(speedups, 2.5), np.percentile(speedups, 97.5))
        
        return {
            'speedup': float(speedup),
            'speedup_std': float(np.std(speedups)),
            'speedup_ci': (float(speedup_ci[0]), float(speedup_ci[1])),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'orig_mean': float(orig_times.mean()),
            'triton_mean': float(triton_times.mean()),
            'orig_std': float(orig_times.std()),
            'triton_std': float(triton_times.std()),
        }


def test_correctness(device: str = 'cuda', atol: float = 1e-5, rtol: float = 1e-4):
    """Test that Triton implementation matches original."""
    print("Testing correctness...")
    
    batch_sizes = [1, 2, 4]
    seq_lens = [512, 768, 1024]
    dims = [256, 512, 1024]
    k_values = [2, 3, 4]
    
    from pom_triton import (
        po2_triton, po3_triton, po4_triton,
        polynomial_aggregation_triton, polynomial_selection_triton
    )
    
    all_passed = True
    
    for batch in batch_sizes:
        for seq_len in seq_lens:
            for dim in dims:
                for k in k_values:
                    # Test polynomial functions
                    x = torch.randn(batch, seq_len, dim, device=device)
                    coeff = torch.randn(dim, k, device=device)
                    
                    # Test polynomial expansion
                    if k == 2:
                        orig = po2(x, coeff)
                        triton = po2_triton(x, coeff)
                    elif k == 3:
                        orig = po3(x, coeff)
                        triton = po3_triton(x, coeff)
                    elif k == 4:
                        orig = po4(x, coeff)
                        triton = po4_triton(x, coeff)
                    
                    match = torch.allclose(orig, triton, atol=atol, rtol=rtol)
                    if not match:
                        print(f"❌ po{k} mismatch: batch={batch}, seq={seq_len}, dim={dim}")
                        print(f"   Max diff: {(orig - triton).abs().max().item()}")
                        print(f"   Orig: {orig[0, :3]}")
                        print(f"   Triton: {triton[0, :3]}")
                        all_passed = False
                    else:
                        print(f"✓ po{k}: batch={batch}, seq={seq_len}, dim={dim}")
    
    print()
    if all_passed:
        print("✅ All correctness tests passed!")
    else:
        print("❌ Some correctness tests failed!")
    
    return all_passed


def test_backward_correctness(device: str = 'cuda', atol: float = 1e-4, rtol: float = 1e-3):
    """Test that gradients computed by Triton match original implementation."""
    print("\nTesting backward pass correctness...")
    
    from pom_triton import (
        po2_triton, po3_triton, po4_triton
    )
    
    batch_sizes = [1, 2]
    seq_lens = [256, 512]
    dims = [128, 256]
    k_values = [2, 3, 4]
    
    all_passed = True
    
    for batch in batch_sizes:
        for seq_len in seq_lens:
            for dim in dims:
                for k in k_values:
                    # Test polynomial functions with gradients
                    x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
                    coeff = torch.randn(dim, k, device=device, requires_grad=True)
                    
                    # Detach for separate gradient computation
                    x_orig = x.detach().clone().requires_grad_(True)
                    x_triton = x.detach().clone().requires_grad_(True)
                    coeff_orig = coeff.detach().clone().requires_grad_(True)
                    coeff_triton = coeff.detach().clone().requires_grad_(True)
                    
                    # Forward pass
                    if k == 2:
                        out_orig = po2(x_orig, coeff_orig)
                        out_triton = po2_triton(x_triton, coeff_triton)
                    elif k == 3:
                        out_orig = po3(x_orig, coeff_orig)
                        out_triton = po3_triton(x_triton, coeff_triton)
                    elif k == 4:
                        out_orig = po4(x_orig, coeff_orig)
                        out_triton = po4_triton(x_triton, coeff_triton)
                    
                    # Compute a simple loss and backward
                    loss_orig = out_orig.sum()
                    loss_triton = out_triton.sum()
                    
                    loss_orig.backward()
                    loss_triton.backward()
                    
                    # Check gradients match
                    if x_orig.grad is not None and x_triton.grad is not None:
                        x_grad_match = torch.allclose(x_orig.grad, x_triton.grad, atol=atol, rtol=rtol)
                        if not x_grad_match:
                            print(f"❌ po{k} x gradient mismatch: batch={batch}, seq={seq_len}, dim={dim}")
                            print(f"   Max diff: {(x_orig.grad - x_triton.grad).abs().max().item()}")
                            all_passed = False
                    
                    if coeff_orig.grad is not None and coeff_triton.grad is not None:
                        coeff_grad_match = torch.allclose(coeff_orig.grad, coeff_triton.grad, atol=atol, rtol=rtol)
                        if not coeff_grad_match:
                            print(f"❌ po{k} coeff gradient mismatch: batch={batch}, seq={seq_len}, dim={dim}")
                            print(f"   Max diff: {(coeff_orig.grad - coeff_triton.grad).abs().max().item()}")
                            all_passed = False
                    
                    if x_grad_match and coeff_grad_match:
                        print(f"✓ po{k} backward: batch={batch}, seq={seq_len}, dim={dim}")
    
    print()
    if all_passed:
        print("✅ All backward pass tests passed!")
    else:
        print("❌ Some backward pass tests failed!")
    
    return all_passed


def benchmark_backward(
    n_warmup: int = 3,
    n_runs: int = 10,
    device: str = 'cuda'
) -> Dict:
    """Benchmark backward pass performance."""
    print("\nBenchmarking backward passes...")
    
    from pom_triton import po3_triton
    
    results = {}
    
    seq_lens = [512, 768, 1024]
    dims = [256, 512, 1024]
    k = 3
    batch = 4
    
    for seq_len in seq_lens:
        for dim in dims:
            key = f"{seq_len}_{dim}"
            results[key] = {'original': [], 'triton': []}
            
            # Test with gradient computation
            x = torch.randn(batch, seq_len, dim, device=device)
            coeff = torch.randn(dim, k, device=device)
            
            # Warmup - original
            for _ in range(n_warmup):
                x_copy = x.detach().clone().requires_grad_(True)
                coeff_copy = coeff.detach().clone().requires_grad_(True)
                out = po3(x_copy, coeff_copy)
                loss = out.sum()
                loss.backward()
            
            # Warmup - triton
            for _ in range(n_warmup):
                x_copy = x.detach().clone().requires_grad_(True)
                coeff_copy = coeff.detach().clone().requires_grad_(True)
                out = po3_triton(x_copy, coeff_copy)
                loss = out.sum()
                loss.backward()
            
            torch.cuda.synchronize()
            
            # Benchmark original backward
            times = []
            for _ in range(n_runs):
                x_copy = x.detach().clone().requires_grad_(True)
                coeff_copy = coeff.detach().clone().requires_grad_(True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = po3(x_copy, coeff_copy)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            results[key]['original'] = times
            
            # Benchmark Triton backward
            times = []
            for _ in range(n_runs):
                x_copy = x.detach().clone().requires_grad_(True)
                coeff_copy = coeff.detach().clone().requires_grad_(True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = po3_triton(x_copy, coeff_copy)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            results[key]['triton'] = times
            
            # Compute speedup
            orig_mean = np.mean(results[key]['original'])
            triton_mean = np.mean(results[key]['triton'])
            speedup = orig_mean / triton_mean
            
            orig_times = np.array(results[key]['original'])
            triton_times = np.array(results[key]['triton'])
            t_stat, p_value = stats.ttest_ind(orig_times, triton_times)
            
            print(f"seq={seq_len}, dim={dim}: "
                  f"speedup={speedup:.2f}x "
                  f"(orig={orig_mean:.2f}ms, triton={triton_mean:.2f}ms, "
                  f"p={p_value:.2e}, sig={'✓' if p_value < 0.05 else '✗'})")
            
            results[key]['speedup'] = speedup
            results[key]['p_value'] = p_value
            results[key]['significant'] = p_value < 0.05
    
    return results


def print_backward_summary(results: Dict):
    """Print backward benchmark summary."""
    print(f"\n{'=' * 70}")
    print(f"Backward Pass Benchmarks")
    print(f"{'=' * 70}")
    print(f"{'Config':<20} {'Speedup':<12} {'P-Value':<15} {'Sig?':<8}")
    print(f"{'-' * 70}")
    
    speedups = []
    significant_count = 0
    
    for key in sorted(results.keys()):
        speedup = results[key]['speedup']
        p_value = results[key]['p_value']
        is_sig = results[key]['significant']
        
        speedups.append(speedup)
        if is_sig:
            significant_count += 1
        
        print(f"{key:<20} "
              f"{speedup:>6.2f}x      "
              f"{p_value:>10.2e}    "
              f"{'✓' if is_sig else '✗':<8}")
    
    if speedups:
        print(f"{'-' * 70}")
        print(f"{'Average':<20} {np.mean(speedups):>6.2f}x")
        print(f"{'Speedup Std Dev':<20} {np.std(speedups):>6.2f}x")
        print(f"{'Significant Cases':<20} {significant_count}/{len(speedups)}")
        print(f"{'Overall Speedup':<20} {'YES ✓' if np.mean(speedups) > 1.0 else 'NO ✗'}")


def benchmark_polynomial_ops(
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = 'cuda'
) -> BenchmarkResults:
    """Benchmark polynomial expand operations."""
    print("Benchmarking polynomial operations...")
    
    results = BenchmarkResults()
    
    seq_lens = [512, 768, 1024]
    dims = [256, 512, 1024]
    k = 3  # Use degree 3 polynomial
    batch = 4
    
    from pom_triton import po3_triton
    
    for seq_len in seq_lens:
        for dim in dims:
            x = torch.randn(batch, seq_len, dim, device=device)
            coeff = torch.randn(dim, k, device=device)
            
            # Warmup
            for _ in range(n_warmup):
                _ = po3(x, coeff)
                _ = po3_triton(x, coeff)
            
            torch.cuda.synchronize()
            
            # Original implementation
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = po3(x, coeff)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            for t in times:
                results.add_result(seq_len, dim, t, is_triton=False)
            
            # Triton implementation
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = po3_triton(x, coeff)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            for t in times:
                results.add_result(seq_len, dim, t, is_triton=True)
            
            # Print progress
            speedup_info = results.compute_speedup(seq_len, dim)
            if speedup_info:
                print(f"seq={seq_len}, dim={dim}: "
                      f"speedup={speedup_info['speedup']:.2f}x "
                      f"(p={speedup_info['p_value']:.2e}, "
                      f"sig={'✓' if speedup_info['significant'] else '✗'})")
    
    return results


def benchmark_aggregation(
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = 'cuda'
) -> BenchmarkResults:
    """Benchmark polynomial aggregation operations."""
    print("\nBenchmarking polynomial aggregation...")
    
    results = BenchmarkResults()
    
    seq_lens = [512, 768, 1024]
    dims = [256, 512, 1024]  # These are the expanded dims
    k = 3
    batch = 4
    
    from pom_triton import polynomial_aggregation_triton
    
    for seq_len in seq_lens:
        for dim in dims:
            # For aggregation, x and coeff both use expanded dimensions
            xc = torch.randn(batch, seq_len, dim, device=device)
            coeff = torch.randn(dim, k, device=device)  # coeff matches input dim
            
            # Warmup
            for _ in range(n_warmup):
                _ = polynomial_aggregation_(xc, coeff, k, mask=None)
                _ = polynomial_aggregation_triton(xc, coeff, k, mask=None)
            
            torch.cuda.synchronize()
            
            # Original
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = polynomial_aggregation_(xc, coeff, k, mask=None)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            for t in times:
                results.add_result(seq_len, dim, t, is_triton=False)
            
            # Triton
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = polynomial_aggregation_triton(xc, coeff, k, mask=None)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            for t in times:
                results.add_result(seq_len, dim, t, is_triton=True)
            
            speedup_info = results.compute_speedup(seq_len, dim)
            if speedup_info:
                print(f"seq={seq_len}, dim={dim}: "
                      f"speedup={speedup_info['speedup']:.2f}x "
                      f"(p={speedup_info['p_value']:.2e}, "
                      f"sig={'✓' if speedup_info['significant'] else '✗'})")
    
    return results


def print_summary(results: BenchmarkResults, name: str):
    """Print benchmark summary with statistics."""
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    print(f"{'Config':<20} {'Speedup':<12} {'P-Value':<15} {'Sig?':<8}")
    print(f"{'-' * 70}")
    
    all_speedups = []
    all_significant = []
    
    for key in sorted(results.results.keys()):
        speedup_info = results.compute_speedup(int(key.split('_')[0]), int(key.split('_')[1]))
        if speedup_info:
            all_speedups.append(speedup_info['speedup'])
            all_significant.append(speedup_info['significant'])
            
            print(f"{key:<20} "
                  f"{speedup_info['speedup']:>6.2f}x      "
                  f"{speedup_info['p_value']:>10.2e}    "
                  f"{'✓' if speedup_info['significant'] else '✗':<8}")
    
    if all_speedups:
        print(f"{'-' * 70}")
        print(f"{'Average':<20} {np.mean(all_speedups):>6.2f}x")
        print(f"{'Speedup Std Dev':<20} {np.std(all_speedups):>6.2f}x")
        print(f"{'Significant Cases':<20} {sum(all_significant)}/{len(all_significant)}")
        print(f"{'Overall Speedup':<20} {'YES ✓' if np.mean(all_speedups) > 1.0 else 'NO ✗'}")


def main():
    """Run all tests and benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--correctness', action='store_true', help='Run forward correctness tests')
    parser.add_argument('--backward', action='store_true', help='Run backward correctness tests')
    parser.add_argument('--benchmark', action='store_true', help='Run forward benchmarks')
    parser.add_argument('--benchmark-backward', action='store_true', help='Run backward benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--save-results', type=str, help='Save results to JSON file')
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.all or (not args.correctness and not args.backward and not args.benchmark and not args.benchmark_backward):
        args.correctness = True
        args.backward = True
        args.benchmark = True
        args.benchmark_backward = True
    
    if args.correctness:
        try:
            passed = test_correctness(device=args.device)
            if not passed:
                print("Forward correctness tests failed. Fix issues before continuing.")
                return
        except Exception as e:
            print(f"Error running forward correctness tests: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if args.backward:
        try:
            passed = test_backward_correctness(device=args.device)
            if not passed:
                print("⚠️  Backward tests failed. There may be gradient computation issues.")
                # Don't return - continue to allow analysis
        except Exception as e:
            print(f"Error running backward correctness tests: {e}")
            import traceback
            traceback.print_exc()
    
    if args.benchmark:
        try:
            poly_results = benchmark_polynomial_ops(device=args.device)
            agg_results = benchmark_aggregation(device=args.device)
            
            print_summary(poly_results, "Polynomial Expansion Benchmarks")
            print_summary(agg_results, "Aggregation Benchmarks")
            
            if args.save_results:
                save_results_to_json(poly_results, agg_results, args.save_results)
                print(f"\nResults saved to {args.save_results}")
        except Exception as e:
            print(f"Error running benchmarks: {e}")
            import traceback
            traceback.print_exc()
    
    if args.benchmark_backward:
        try:
            backward_results = benchmark_backward(device=args.device)
            print_backward_summary(backward_results)
        except Exception as e:
            print(f"Error running backward benchmarks: {e}")
            import traceback
            traceback.print_exc()


def save_results_to_json(poly_results: BenchmarkResults, 
                         agg_results: BenchmarkResults,
                         filepath: str):
    """Save benchmark results to JSON file."""
    output = {
        'polynomial': {},
        'aggregation': {}
    }
    
    for key in poly_results.results.keys():
        seq_len, dim = int(key.split('_')[0]), int(key.split('_')[1])
        speedup_info = poly_results.compute_speedup(seq_len, dim)
        if speedup_info:
            # Convert boolean to string for JSON serialization
            speedup_info['significant'] = str(speedup_info['significant'])
            output['polynomial'][key] = speedup_info
    
    for key in agg_results.results.keys():
        seq_len, dim = int(key.split('_')[0]), int(key.split('_')[1])
        speedup_info = agg_results.compute_speedup(seq_len, dim)
        if speedup_info:
            # Convert boolean to string for JSON serialization
            speedup_info['significant'] = str(speedup_info['significant'])
            output['aggregation'][key] = speedup_info
    
    Path(filepath).write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
