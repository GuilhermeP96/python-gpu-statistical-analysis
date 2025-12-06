"""
GPU Statistics Module

Provides GPU-accelerated statistical operations with automatic CPU fallback.
Uses CuPy for NVIDIA CUDA GPUs and falls back to NumPy when unavailable.
"""

import time
from typing import Dict, Any, Optional, List
import numpy as np

# GPU detection
GPU_AVAILABLE = False
DEVICE_NAME = "CPU"

try:
    import cupy as cp
    if cp.cuda.is_available():
        GPU_AVAILABLE = True
        device = cp.cuda.Device(0)
        DEVICE_NAME = device.compute_capability
except ImportError:
    pass


class GPUStatistics:
    """
    Statistical operations with GPU acceleration.
    
    Automatically detects GPU availability and falls back to CPU.
    Provides Monte Carlo simulations, frequency analysis, and benchmarking.
    
    Attributes:
        gpu_available: Whether GPU is available
        device_name: Name of the compute device
        batch_size: Batch size for GPU operations
    """
    
    def __init__(self, batch_size: int = 50000, memory_pool: bool = True):
        """
        Initialize GPU statistics module.
        
        Args:
            batch_size: Batch size for GPU operations (reduce if out of memory)
            memory_pool: Enable CuPy memory pooling for better performance
        """
        self.gpu_available = GPU_AVAILABLE
        self.device_name = DEVICE_NAME
        self.batch_size = batch_size
        self._memory_pool = memory_pool
        
        if self.gpu_available and memory_pool:
            self._setup_memory_pool()
    
    def _setup_memory_pool(self) -> None:
        """Configure CuPy memory pool for efficient allocation."""
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.8)
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the compute device.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "gpu_available": self.gpu_available,
            "device": "CPU (NumPy)",
            "compute_capability": None,
            "total_memory": None,
            "free_memory": None,
        }
        
        if self.gpu_available:
            device = cp.cuda.Device(0)
            info["device"] = f"GPU (CuPy {cp.__version__})"
            info["compute_capability"] = device.compute_capability
            info["total_memory"] = device.mem_info[1]
            info["free_memory"] = device.mem_info[0]
        
        return info
    
    def monte_carlo(
        self, 
        n_simulations: int = 500000,
        n_items: int = 60,
        sample_size: int = 6,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Generates random samples and computes frequency distribution.
        Uses GPU when available, falls back to CPU otherwise.
        
        Args:
            n_simulations: Number of simulations to run
            n_items: Total number of items to sample from
            sample_size: Number of items per sample
            batch_size: Override default batch size
            
        Returns:
            Dictionary with simulation results
        """
        if self.gpu_available:
            return self._monte_carlo_gpu(
                n_simulations, n_items, sample_size, 
                batch_size or self.batch_size
            )
        return self._monte_carlo_cpu(n_simulations, n_items, sample_size)
    
    def _monte_carlo_gpu(
        self, 
        n_simulations: int, 
        n_items: int, 
        sample_size: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """GPU-accelerated Monte Carlo simulation."""
        start = time.perf_counter()
        
        n_batches = (n_simulations + batch_size - 1) // batch_size
        total_freq = cp.zeros(n_items, dtype=cp.int64)
        
        for batch_idx in range(n_batches):
            current_batch = min(batch_size, n_simulations - batch_idx * batch_size)
            
            # Generate random samples using GPU
            random_matrix = cp.random.random((current_batch, n_items))
            sorted_indices = cp.argsort(random_matrix, axis=1)[:, :sample_size]
            samples = sorted_indices + 1
            
            # Count frequencies
            for i in range(n_items):
                total_freq[i] += cp.sum(samples == (i + 1))
            
            # Free GPU memory
            del random_matrix, sorted_indices, samples
            cp.get_default_memory_pool().free_all_blocks()
        
        frequencies = cp.asnumpy(total_freq)
        elapsed = time.perf_counter() - start
        
        return {
            "method": "gpu",
            "simulations": n_simulations,
            "elapsed": elapsed,
            "rate": n_simulations / elapsed,
            "frequencies": frequencies.tolist(),
            "batches": n_batches,
        }
    
    def _monte_carlo_cpu(
        self, 
        n_simulations: int, 
        n_items: int, 
        sample_size: int
    ) -> Dict[str, Any]:
        """CPU-based Monte Carlo simulation."""
        start = time.perf_counter()
        
        frequencies = np.zeros(n_items, dtype=np.int64)
        
        for _ in range(n_simulations):
            sample = np.random.choice(n_items, size=sample_size, replace=False) + 1
            for item in sample:
                frequencies[item - 1] += 1
        
        elapsed = time.perf_counter() - start
        
        return {
            "method": "cpu",
            "simulations": n_simulations,
            "elapsed": elapsed,
            "rate": n_simulations / elapsed,
            "frequencies": frequencies.tolist(),
            "batches": 1,
        }
    
    def benchmark(
        self, 
        n_simulations: int = 100000,
        n_items: int = 60,
        sample_size: int = 6
    ) -> Dict[str, Any]:
        """
        Benchmark CPU vs GPU performance.
        
        Args:
            n_simulations: Number of simulations per benchmark
            n_items: Total number of items
            sample_size: Sample size
            
        Returns:
            Dictionary with benchmark results
        """
        # CPU benchmark
        cpu_result = self._monte_carlo_cpu(n_simulations, n_items, sample_size)
        
        result = {
            "cpu": {
                "elapsed": cpu_result["elapsed"],
                "rate": cpu_result["rate"],
            },
            "gpu": None,
            "speedup": 1.0,
        }
        
        # GPU benchmark (if available)
        if self.gpu_available:
            gpu_result = self._monte_carlo_gpu(
                n_simulations, n_items, sample_size, self.batch_size
            )
            result["gpu"] = {
                "elapsed": gpu_result["elapsed"],
                "rate": gpu_result["rate"],
            }
            result["speedup"] = gpu_result["rate"] / cpu_result["rate"]
        
        return result
    
    def frequency_analysis(
        self,
        data: List[List[int]],
        windows: List[int] = None
    ) -> Dict[str, Dict[int, int]]:
        """
        Analyze frequency distribution in sliding windows.
        
        Args:
            data: List of samples (each sample is a list of integers)
            windows: Window sizes for analysis (default: [50, 100, 200])
            
        Returns:
            Dictionary mapping window size to frequency counts
        """
        if windows is None:
            windows = [50, 100, 200]
        
        results = {}
        
        for window in windows:
            if len(data) < window:
                continue
            
            freq = {}
            for sample in data[-window:]:
                for item in sample:
                    freq[item] = freq.get(item, 0) + 1
            
            results[window] = freq
        
        return results
    
    def calculate_delays(
        self,
        data: List[List[int]],
        n_items: int = 60
    ) -> Dict[int, int]:
        """
        Calculate delay (time since last occurrence) for each item.
        
        Args:
            data: List of samples
            n_items: Total number of possible items
            
        Returns:
            Dictionary mapping item to delay count
        """
        delays = {}
        
        for item in range(1, n_items + 1):
            delay = 0
            for sample in reversed(data):
                if item in sample:
                    break
                delay += 1
            delays[item] = delay
        
        return delays
