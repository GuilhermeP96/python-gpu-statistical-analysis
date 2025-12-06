"""
Monte Carlo Simulation Example

Demonstrates GPU-accelerated Monte Carlo simulation with automatic CPU fallback.
"""

from src.gpu_stats import GPUStatistics


def main():
    # Initialize statistics module
    stats = GPUStatistics(batch_size=50000)
    
    # Print device information
    device_info = stats.get_device_info()
    print("Device Information:")
    print(f"  GPU Available: {device_info['gpu_available']}")
    print(f"  Device: {device_info['device']}")
    if device_info['compute_capability']:
        print(f"  Compute Capability: {device_info['compute_capability']}")
    print()
    
    # Run Monte Carlo simulation
    print("Running Monte Carlo simulation...")
    result = stats.monte_carlo(
        n_simulations=500000,
        n_items=60,
        sample_size=6
    )
    
    print(f"\nResults:")
    print(f"  Method: {result['method'].upper()}")
    print(f"  Simulations: {result['simulations']:,}")
    print(f"  Elapsed: {result['elapsed']:.2f}s")
    print(f"  Rate: {result['rate']:,.0f} simulations/second")
    print(f"  Batches: {result['batches']}")
    
    # Show frequency distribution
    frequencies = result['frequencies']
    print(f"\nFrequency Distribution (top 10):")
    sorted_freq = sorted(enumerate(frequencies, 1), key=lambda x: x[1], reverse=True)
    for item, freq in sorted_freq[:10]:
        print(f"  Item {item:2d}: {freq:,}")


if __name__ == "__main__":
    main()
