"""
Weight Optimizer Module

Dynamic weight optimization using backtesting with sliding window validation.
Finds optimal weights for multi-factor scoring systems.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


class WeightOptimizer:
    """
    Optimize scoring weights using historical data validation.
    
    Uses sliding window backtesting to find weights that maximize
    predictive accuracy on recent data.
    
    Attributes:
        data: Historical data for optimization
        history_path: Path to optimization history file
    """
    
    def __init__(
        self, 
        data: List[List[int]],
        history_path: Optional[Path] = None
    ):
        """
        Initialize weight optimizer.
        
        Args:
            data: Historical data (list of samples)
            history_path: Path to save optimization history
        """
        self.data = data
        self.history_path = history_path or Path("optimization_history.json")
        self._history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load optimization history from file."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"optimizations": []}
    
    def _save_history(self, weights: Dict[str, float], score: float) -> None:
        """Save optimization result to history."""
        self._history["optimizations"].append({
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(self.data),
            "weights": weights,
            "validation_score": score,
        })
        
        # Keep last 50 optimizations
        if len(self._history["optimizations"]) > 50:
            self._history["optimizations"] = self._history["optimizations"][-50:]
        
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._history["optimizations"]
    
    def get_last_optimization(self) -> Optional[Dict[str, Any]]:
        """Get most recent optimization result."""
        if self._history["optimizations"]:
            return self._history["optimizations"][-1]
        return None
    
    def optimize(
        self,
        train_ratio: float = 0.8,
        weight_components: List[str] = None,
        weight_ranges: Dict[str, List[float]] = None,
        top_sizes: List[int] = None
    ) -> Dict[str, float]:
        """
        Find optimal weights using grid search with backtesting.
        
        Uses sliding window: trains on older data, validates on recent data.
        
        Args:
            train_ratio: Ratio of data to use for training (rest for validation)
            weight_components: Names of weight components
            weight_ranges: Possible values for each weight component
            top_sizes: Top-N sizes for evaluation
            
        Returns:
            Dictionary of optimal weights
        """
        if len(self.data) < 500:
            return self._default_weights(weight_components)
        
        # Default configuration
        if weight_components is None:
            weight_components = ["frequency", "delay", "trend", "random"]
        
        if weight_ranges is None:
            weight_ranges = {
                "frequency": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
                "delay": [0.10, 0.15, 0.20, 0.25, 0.30],
                "trend": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
                "random": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            }
        
        if top_sizes is None:
            top_sizes = [15, 20, 25]
        
        # Split data: train on old, validate on recent
        split_idx = int(len(self.data) * train_ratio)
        train_data = self.data[:split_idx]
        validation_data = self.data[split_idx:]
        
        # Compute training metrics
        train_metrics = self._compute_metrics(train_data)
        
        # Generate weight combinations that sum to 1.0
        candidates = self._generate_weight_combinations(weight_components, weight_ranges)
        
        # Find best weights
        best_weights = None
        best_score = -1
        results = []
        
        for weights in candidates:
            scores = self._compute_scores(train_metrics, weights)
            score = self._evaluate(scores, validation_data, top_sizes)
            results.append({"weights": weights, "score": score})
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
        
        # Save to history
        self._save_history(best_weights, best_score)
        
        return best_weights
    
    def _default_weights(self, components: List[str] = None) -> Dict[str, float]:
        """Return default balanced weights."""
        if components is None:
            components = ["frequency", "delay", "trend", "random"]
        weight = 1.0 / len(components)
        return {c: weight for c in components}
    
    def _generate_weight_combinations(
        self,
        components: List[str],
        ranges: Dict[str, List[float]]
    ) -> List[Dict[str, float]]:
        """Generate all valid weight combinations that sum to 1.0."""
        candidates = []
        
        # Simplified grid search for 4 components
        if len(components) == 4:
            c1, c2, c3, c4 = components
            for w1 in ranges.get(c1, [0.25]):
                for w2 in ranges.get(c2, [0.25]):
                    for w3 in ranges.get(c3, [0.25]):
                        w4 = round(1.0 - w1 - w2 - w3, 2)
                        if w4 in ranges.get(c4, [w4]) or (0.05 <= w4 <= 0.35):
                            candidates.append({
                                c1: w1, c2: w2, c3: w3, c4: w4
                            })
        
        return candidates
    
    def _compute_metrics(self, data: List[List[int]], n_items: int = 60) -> Dict[str, Any]:
        """Compute statistical metrics from training data."""
        # Frequency
        freq = Counter()
        for sample in data:
            freq.update(sample)
        max_freq = max(freq.values()) if freq else 1
        
        # Delays
        delays = {}
        for item in range(1, n_items + 1):
            delay = 0
            for sample in reversed(data):
                if item in sample:
                    break
                delay += 1
            delays[item] = delay
        max_delay = max(delays.values()) if delays else 1
        
        # Recent trend (last 20% of training data)
        window = max(50, len(data) // 5)
        trend = Counter()
        for sample in data[-window:]:
            trend.update(sample)
        max_trend = max(trend.values()) if trend else 1
        
        return {
            "frequency": {item: freq.get(item, 0) / max_freq for item in range(1, n_items + 1)},
            "delay": {item: delays.get(item, 0) / max_delay for item in range(1, n_items + 1)},
            "trend": {item: trend.get(item, 0) / max_trend for item in range(1, n_items + 1)},
            "random": {item: 0.5 for item in range(1, n_items + 1)},
        }
    
    def _compute_scores(
        self, 
        metrics: Dict[str, Dict[int, float]], 
        weights: Dict[str, float]
    ) -> Dict[int, float]:
        """Compute weighted scores for each item."""
        scores = {}
        items = list(metrics["frequency"].keys())
        
        for item in items:
            score = 0
            for component, weight in weights.items():
                if component in metrics:
                    score += metrics[component].get(item, 0) * weight
            scores[item] = score
        
        return scores
    
    def _evaluate(
        self,
        scores: Dict[int, float],
        validation_data: List[List[int]],
        top_sizes: List[int]
    ) -> float:
        """Evaluate scores against validation data."""
        sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        total_score = 0
        for top_size in top_sizes:
            top_n = set(sorted_items[:top_size])
            hits = sum(len(set(sample) & top_n) for sample in validation_data)
            # Normalize by expected hits
            normalized = hits / len(validation_data) / top_size * 6
            total_score += normalized
        
        return total_score / len(top_sizes)
    
    def evaluate(
        self, 
        weights: Dict[str, float], 
        data: List[List[int]] = None
    ) -> float:
        """
        Evaluate a weight configuration.
        
        Args:
            weights: Weight configuration to evaluate
            data: Data to evaluate on (uses self.data if None)
            
        Returns:
            Evaluation score
        """
        if data is None:
            data = self.data
        
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        validation_data = data[split_idx:]
        
        metrics = self._compute_metrics(train_data)
        scores = self._compute_scores(metrics, weights)
        
        return self._evaluate(scores, validation_data, [15, 20, 25])
    
    def compare_with_last(self, current_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare current weights with last optimization.
        
        Args:
            current_weights: Current weight configuration
            
        Returns:
            Comparison results
        """
        last = self.get_last_optimization()
        
        if last is None:
            return {"has_previous": False}
        
        comparison = {
            "has_previous": True,
            "previous_weights": last["weights"],
            "current_weights": current_weights,
            "changes": {},
        }
        
        for key in current_weights:
            prev = last["weights"].get(key, 0)
            curr = current_weights[key]
            comparison["changes"][key] = {
                "previous": prev,
                "current": curr,
                "delta": curr - prev,
            }
        
        return comparison
