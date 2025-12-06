"""
Weight Optimization Example

Demonstrates dynamic weight optimization using backtesting.
"""

import random
from src.weight_optimizer import WeightOptimizer


def generate_sample_data(n_samples: int = 1000, n_items: int = 60, sample_size: int = 6):
    """Generate sample data for demonstration."""
    data = []
    for _ in range(n_samples):
        sample = sorted(random.sample(range(1, n_items + 1), sample_size))
        data.append(sample)
    return data


def main():
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(n_samples=2000)
    print(f"Generated {len(data)} samples\n")
    
    # Initialize optimizer
    optimizer = WeightOptimizer(data)
    
    # Check for previous optimization
    last = optimizer.get_last_optimization()
    if last:
        print(f"Previous optimization found:")
        print(f"  Timestamp: {last['timestamp']}")
        print(f"  Samples: {last['n_samples']}")
        print(f"  Weights: {last['weights']}")
        print()
    
    # Run optimization
    print("Running weight optimization...")
    print("  Training on 80% of data (older samples)")
    print("  Validating on 20% of data (recent samples)")
    print()
    
    optimal_weights = optimizer.optimize(train_ratio=0.8)
    
    print("Optimal Weights Found:")
    for component, weight in optimal_weights.items():
        print(f"  {component}: {weight:.0%}")
    
    # Compare with previous if available
    if last:
        comparison = optimizer.compare_with_last(optimal_weights)
        print("\nComparison with Previous:")
        for component, change in comparison["changes"].items():
            delta = change["delta"] * 100
            sign = "+" if delta > 0 else ""
            print(f"  {component}: {change['previous']:.0%} -> {change['current']:.0%} ({sign}{delta:.0f}pp)")
    
    # Evaluate the weights
    score = optimizer.evaluate(optimal_weights)
    print(f"\nValidation Score: {score:.4f}")


if __name__ == "__main__":
    main()
