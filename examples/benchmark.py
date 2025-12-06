"""
CPU vs GPU Benchmark

Compares performance between CPU (NumPy) and GPU (CuPy) implementations.
"""

from src.gpu_stats import GPUStatistics


def main():
    stats = GPUStatistics()
    
    print("CPU vs GPU Benchmark")
    print("=" * 50)
    
    device_info = stats.get_device_info()
    print(f"\nDevice: {device_info['device']}")
    if device_info['gpu_available']:
        print(f"GPU Memory: {device_info['total_memory'] / 1e9:.1f} GB total")
    print()
    
    # Run benchmarks at different scales
    simulation_counts = [10000, 50000, 100000, 500000]
    
    print(f"{'Simulations':>12} | {'CPU (s)':>10} | {'GPU (s)':>10} | {'Speedup':>10}")
    print("-" * 50)
    
    for n_sims in simulation_counts:
        result = stats.benchmark(n_simulations=n_sims)
        
        cpu_time = result["cpu"]["elapsed"]
        
        if result["gpu"]:
            gpu_time = result["gpu"]["elapsed"]
            speedup = result["speedup"]
            print(f"{n_sims:>12,} | {cpu_time:>10.3f} | {gpu_time:>10.3f} | {speedup:>9.1f}x")
        else:
            print(f"{n_sims:>12,} | {cpu_time:>10.3f} | {'N/A':>10} | {'N/A':>10}")
    
    print()
    
    # Detailed benchmark
    print("Detailed Benchmark (500K simulations)")
    print("-" * 50)
    
    result = stats.benchmark(n_simulations=500000)
    
    print(f"CPU Rate: {result['cpu']['rate']:,.0f} simulations/second")
    if result["gpu"]:
        print(f"GPU Rate: {result['gpu']['rate']:,.0f} simulations/second")
        print(f"Speedup:  {result['speedup']:.1f}x faster with GPU")
    else:
        print("GPU: Not available")


if __name__ == "__main__":
    main()
