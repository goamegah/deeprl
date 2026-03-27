"""
Module d'entrainement et d'evaluation.

Contient:
- Trainer: Boucle d'entrainement
- Evaluator: Evaluation des agents
- Benchmark: Benchmarking complet avec graphiques
"""

from deeprl.training.trainer import Trainer
from deeprl.training.evaluator import Evaluator
from deeprl.training.benchmark import (
    Benchmark,
    BenchmarkSuite,
    AgentBenchmarkResult,
    quick_benchmark
)

__all__ = [
    "Trainer",
    "Evaluator",
    "Benchmark",
    "BenchmarkSuite",
    "AgentBenchmarkResult",
    "quick_benchmark",
]
