"""
Python GPU Statistical Analysis

GPU-accelerated statistical analysis with automatic CPU fallback.
"""

from .gpu_stats import GPUStatistics
from .weight_optimizer import WeightOptimizer

__version__ = "1.0.0"
__all__ = ["GPUStatistics", "WeightOptimizer"]
