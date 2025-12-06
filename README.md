# Python GPU Statistical Analysis

A reference implementation for statistical analysis using GPU acceleration (CuPy/CUDA) with CPU fallback (NumPy).

## Features

- GPU-accelerated Monte Carlo simulations using CuPy
- Automatic CPU fallback when GPU is unavailable
- Dynamic weight optimization via backtesting
- Sliding window validation for time-series data
- Historical optimization tracking

## Performance

Benchmarks on Intel Xeon E5-2676 v3 + NVIDIA GTX 1660 SUPER:

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Monte Carlo (500K) | 14.4s | 0.47s | 30x |
| Rate | 37K sims/s | 1.1M sims/s | - |

## Requirements

### Minimum

- Python 3.11+
- Windows 10/11 or Linux
- 8GB RAM

### GPU Support (Optional)

- NVIDIA GPU with Compute Capability 7.0+
- CUDA Toolkit 12.x
- NVIDIA Driver 525+

## Installation

```bash
git clone https://github.com/GuilhermeP96/python-gpu-statistical-analysis.git
cd python-gpu-statistical-analysis

pip install -r requirements.txt

# Optional: GPU support
pip install cupy-cuda12x==13.6.0
```

## Quick Start

```python
from src.gpu_stats import GPUStatistics

# Initialize with automatic GPU/CPU detection
stats = GPUStatistics()

# Run Monte Carlo simulation
results = stats.monte_carlo(n_simulations=500000)

# Get performance metrics
print(f"Method: {results['method']}")
print(f"Rate: {results['rate']:,.0f} simulations/second")
```

## Project Structure

```
python-gpu-statistical-analysis/
├── src/
│   ├── __init__.py
│   ├── gpu_stats.py          # Core GPU/CPU statistics module
│   ├── weight_optimizer.py   # Dynamic weight optimization
│   └── backtesting.py        # Backtesting framework
├── examples/
│   ├── monte_carlo.py        # Monte Carlo simulation example
│   ├── optimization.py       # Weight optimization example
│   └── benchmark.py          # CPU vs GPU benchmark
├── tests/
│   └── test_gpu_stats.py     # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Usage

### Monte Carlo Simulation

```python
from src.gpu_stats import GPUStatistics

stats = GPUStatistics()

# GPU-accelerated Monte Carlo
result = stats.monte_carlo(
    n_simulations=1000000,
    batch_size=100000
)

print(f"Elapsed: {result['elapsed']:.2f}s")
print(f"Rate: {result['rate']:,.0f} sims/s")
```

### Weight Optimization with Backtesting

```python
from src.weight_optimizer import WeightOptimizer

optimizer = WeightOptimizer(data)

# Find optimal weights using historical validation
weights = optimizer.optimize(
    train_ratio=0.8,
    validation_ratio=0.2
)

print(f"Optimal weights: {weights}")
```

### CPU/GPU Benchmark

```python
from src.gpu_stats import GPUStatistics

stats = GPUStatistics()
benchmark = stats.benchmark(n_simulations=500000)

print(f"CPU: {benchmark['cpu']['rate']:,.0f} sims/s")
print(f"GPU: {benchmark['gpu']['rate']:,.0f} sims/s")
print(f"Speedup: {benchmark['speedup']:.1f}x")
```

## Performance

Benchmarks on Intel Xeon E5-2676 v3 + NVIDIA GTX 1660 SUPER:

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Monte Carlo (500K) | 14.4s | 0.47s | 30x |
| Rate | 37K sims/s | 1.1M sims/s | - |

## Configuration

### GPU Memory Management

```python
stats = GPUStatistics(
    batch_size=50000,      # Reduce if out of memory
    memory_pool=True       # Enable memory pooling
)
```

### CuPy Configuration

```python
import cupy as cp

# Check available memory
mempool = cp.get_default_memory_pool()
print(f"Used: {mempool.used_bytes() / 1e9:.2f} GB")

# Clear memory
mempool.free_all_blocks()
```

## Troubleshooting

### CuPy Installation Issues

```bash
# Verify CUDA version
nvcc --version

# Install matching CuPy version
pip uninstall cupy-cuda12x
pip install cupy-cuda12x
```

### Out of Memory Errors

Reduce batch size in GPU operations:

```python
stats = GPUStatistics(batch_size=25000)
```

### No GPU Detected

The system automatically falls back to CPU. Verify GPU availability:

```python
from src.gpu_stats import GPUStatistics

stats = GPUStatistics()
print(f"GPU available: {stats.gpu_available}")
print(f"Device: {stats.device_name}")
```

## API Reference

### GPUStatistics

| Method | Description |
|--------|-------------|
| `monte_carlo(n_simulations, batch_size)` | Run Monte Carlo simulation |
| `benchmark(n_simulations)` | Compare CPU vs GPU performance |
| `get_device_info()` | Get GPU device information |

### WeightOptimizer

| Method | Description |
|--------|-------------|
| `optimize(train_ratio, validation_ratio)` | Find optimal weights |
| `evaluate(weights, data)` | Evaluate weight configuration |
| `get_history()` | Get optimization history |

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
