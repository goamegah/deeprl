"""
DeepRL - Bibliothèque pédagogique de Deep Reinforcement Learning

Cette bibliothèque implémente les algorithmes de RL de manière claire et compréhensible.

Modules:
- envs: Environnements (LineWorld, GridWorld, TicTacToe, Quarto)
- agents: Agents (Random, Human, Q-Learning)
- training: Entraînement et évaluation
- gui: Interface graphique Pygame
"""

__version__ = "0.2.0"
__author__ = "DeepRL Team"

# Imports pratiques
from deeprl.envs import (
    LineWorld, GridWorld, TicTacToe, TicTacToeVsRandom,
    Quarto, QuartoVsRandom
)
from deeprl.agents import (
    Agent, RandomAgent, HumanAgent, TabularQLearning
)
from deeprl.training import Trainer, Evaluator
