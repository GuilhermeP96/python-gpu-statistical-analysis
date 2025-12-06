"""
Backtesting Module

Framework for validating statistical strategies using historical data.
Implements sliding window validation with temporal isolation.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from collections import Counter


class Backtester:
    """
    Backtesting framework for statistical strategies.
    
    Validates strategies using sliding window approach where
    future data is hidden during training/prediction.
    
    Attributes:
        data: Historical data for backtesting
        results: List of backtest results
    """
    
    def __init__(self, data: List[List[int]]):
        """
        Initialize backtester.
        
        Args:
            data: Historical data (list of samples, chronologically ordered)
        """
        self.data = data
        self.results = []
    
    def run(
        self,
        strategy: Callable[[List[List[int]]], List[int]],
        start_idx: int = None,
        end_idx: int = None,
        window_size: int = 500
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Function that takes historical data and returns prediction
            start_idx: Starting index for backtest
            end_idx: Ending index for backtest
            window_size: Size of historical window for strategy
            
        Returns:
            Backtest results
        """
        if start_idx is None:
            start_idx = window_size
        if end_idx is None:
            end_idx = len(self.data)
        
        hits = {4: 0, 5: 0, 6: 0}
        total_tests = 0
        
        for i in range(start_idx, end_idx):
            # Get historical data up to current point
            historical = self.data[max(0, i - window_size):i]
            
            if len(historical) < window_size // 2:
                continue
            
            # Get strategy prediction
            prediction = strategy(historical)
            
            # Compare with actual result
            actual = set(self.data[i])
            predicted = set(prediction)
            matches = len(actual & predicted)
            
            if matches >= 4:
                hits[matches] = hits.get(matches, 0) + 1
            
            total_tests += 1
        
        result = {
            "strategy": strategy.__name__ if hasattr(strategy, "__name__") else "custom",
            "total_tests": total_tests,
            "hits": hits,
            "hit_rate_4": hits[4] / total_tests if total_tests > 0 else 0,
            "hit_rate_5": hits[5] / total_tests if total_tests > 0 else 0,
            "hit_rate_6": hits[6] / total_tests if total_tests > 0 else 0,
            "window_size": window_size,
        }
        
        self.results.append(result)
        return result
    
    def compare_strategies(
        self,
        strategies: List[Callable[[List[List[int]]], List[int]]],
        window_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple strategies.
        
        Args:
            strategies: List of strategy functions
            window_size: Historical window size
            
        Returns:
            List of results for each strategy
        """
        results = []
        for strategy in strategies:
            result = self.run(strategy, window_size=window_size)
            results.append(result)
        
        # Sort by hit rate
        results.sort(key=lambda x: x["hit_rate_4"], reverse=True)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all backtest results."""
        if not self.results:
            return {"total_backtests": 0}
        
        return {
            "total_backtests": len(self.results),
            "best_hit_rate_4": max(r["hit_rate_4"] for r in self.results),
            "average_hit_rate_4": sum(r["hit_rate_4"] for r in self.results) / len(self.results),
            "results": self.results,
        }
    
    def save_results(self, path: Path) -> None:
        """Save backtest results to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.results,
            }, f, indent=2)
    
    @staticmethod
    def create_frequency_strategy(top_n: int = 20, sample_size: int = 6):
        """
        Create a frequency-based strategy.
        
        Args:
            top_n: Number of top items to consider
            sample_size: Size of output sample
            
        Returns:
            Strategy function
        """
        def strategy(historical: List[List[int]]) -> List[int]:
            freq = Counter()
            for sample in historical:
                freq.update(sample)
            
            top_items = [item for item, _ in freq.most_common(top_n)]
            
            import random
            return sorted(random.sample(top_items, min(sample_size, len(top_items))))
        
        strategy.__name__ = f"frequency_top{top_n}"
        return strategy
    
    @staticmethod
    def create_delay_strategy(top_n: int = 20, sample_size: int = 6, n_items: int = 60):
        """
        Create a delay-based strategy (items not seen recently).
        
        Args:
            top_n: Number of top delayed items to consider
            sample_size: Size of output sample
            n_items: Total number of possible items
            
        Returns:
            Strategy function
        """
        def strategy(historical: List[List[int]]) -> List[int]:
            delays = {}
            for item in range(1, n_items + 1):
                delay = 0
                for sample in reversed(historical):
                    if item in sample:
                        break
                    delay += 1
                delays[item] = delay
            
            sorted_items = sorted(delays.keys(), key=lambda x: delays[x], reverse=True)
            top_items = sorted_items[:top_n]
            
            import random
            return sorted(random.sample(top_items, min(sample_size, len(top_items))))
        
        strategy.__name__ = f"delay_top{top_n}"
        return strategy
    
    @staticmethod
    def create_mixed_strategy(
        freq_weight: float = 0.5,
        delay_weight: float = 0.5,
        top_n: int = 20,
        sample_size: int = 6,
        n_items: int = 60
    ):
        """
        Create a mixed strategy combining frequency and delay.
        
        Args:
            freq_weight: Weight for frequency component
            delay_weight: Weight for delay component
            top_n: Number of top items to consider
            sample_size: Size of output sample
            n_items: Total number of possible items
            
        Returns:
            Strategy function
        """
        def strategy(historical: List[List[int]]) -> List[int]:
            # Frequency
            freq = Counter()
            for sample in historical:
                freq.update(sample)
            max_freq = max(freq.values()) if freq else 1
            
            # Delays
            delays = {}
            for item in range(1, n_items + 1):
                delay = 0
                for sample in reversed(historical):
                    if item in sample:
                        break
                    delay += 1
                delays[item] = delay
            max_delay = max(delays.values()) if delays else 1
            
            # Combined score
            scores = {}
            for item in range(1, n_items + 1):
                freq_score = freq.get(item, 0) / max_freq
                delay_score = delays.get(item, 0) / max_delay
                scores[item] = freq_score * freq_weight + delay_score * delay_weight
            
            sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            top_items = sorted_items[:top_n]
            
            import random
            return sorted(random.sample(top_items, min(sample_size, len(top_items))))
        
        strategy.__name__ = f"mixed_f{freq_weight}_d{delay_weight}"
        return strategy
