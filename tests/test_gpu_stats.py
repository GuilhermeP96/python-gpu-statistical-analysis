"""
Unit Tests for GPU Statistics Module
"""

import pytest
from src.gpu_stats import GPUStatistics


class TestGPUStatistics:
    """Tests for GPUStatistics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GPUStatistics(batch_size=10000)
    
    def test_initialization(self):
        """Test module initialization."""
        assert self.stats is not None
        assert isinstance(self.stats.gpu_available, bool)
        assert isinstance(self.stats.batch_size, int)
    
    def test_get_device_info(self):
        """Test device info retrieval."""
        info = self.stats.get_device_info()
        
        assert "gpu_available" in info
        assert "device" in info
        assert isinstance(info["gpu_available"], bool)
    
    def test_monte_carlo_cpu(self):
        """Test Monte Carlo simulation on CPU."""
        result = self.stats._monte_carlo_cpu(
            n_simulations=1000,
            n_items=60,
            sample_size=6
        )
        
        assert result["method"] == "cpu"
        assert result["simulations"] == 1000
        assert len(result["frequencies"]) == 60
        assert result["elapsed"] > 0
        assert result["rate"] > 0
    
    def test_monte_carlo_auto(self):
        """Test automatic GPU/CPU selection."""
        result = self.stats.monte_carlo(n_simulations=1000)
        
        assert result["method"] in ["cpu", "gpu"]
        assert result["simulations"] == 1000
        assert len(result["frequencies"]) == 60
    
    def test_frequency_distribution(self):
        """Test that frequencies sum correctly."""
        result = self.stats.monte_carlo(n_simulations=10000)
        
        total = sum(result["frequencies"])
        expected = 10000 * 6  # 6 items per simulation
        
        assert total == expected
    
    def test_benchmark(self):
        """Test benchmark function."""
        result = self.stats.benchmark(n_simulations=1000)
        
        assert "cpu" in result
        assert "gpu" in result
        assert "speedup" in result
        assert result["cpu"]["rate"] > 0
    
    def test_frequency_analysis(self):
        """Test frequency analysis with windows."""
        data = [[1, 2, 3, 4, 5, 6] for _ in range(200)]
        result = self.stats.frequency_analysis(data, windows=[50, 100])
        
        assert 50 in result
        assert 100 in result
        assert result[50][1] == 50
    
    def test_calculate_delays(self):
        """Test delay calculation."""
        data = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6],
        ]
        delays = self.stats.calculate_delays(data, n_items=12)
        
        assert delays[1] == 0  # In last sample
        assert delays[7] == 1  # Not in last sample
        assert delays[60] == 3  # Never appeared


class TestWeightOptimizer:
    """Tests for WeightOptimizer class."""
    
    def test_import(self):
        """Test module import."""
        from src.weight_optimizer import WeightOptimizer
        assert WeightOptimizer is not None
    
    def test_default_weights(self):
        """Test default weight generation."""
        from src.weight_optimizer import WeightOptimizer
        
        data = [[1, 2, 3, 4, 5, 6] for _ in range(100)]
        optimizer = WeightOptimizer(data)
        
        weights = optimizer._default_weights()
        
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestBacktester:
    """Tests for Backtester class."""
    
    def test_import(self):
        """Test module import."""
        from src.backtesting import Backtester
        assert Backtester is not None
    
    def test_strategy_creation(self):
        """Test strategy factory methods."""
        from src.backtesting import Backtester
        
        freq_strategy = Backtester.create_frequency_strategy(top_n=20)
        delay_strategy = Backtester.create_delay_strategy(top_n=20)
        mixed_strategy = Backtester.create_mixed_strategy()
        
        assert callable(freq_strategy)
        assert callable(delay_strategy)
        assert callable(mixed_strategy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
