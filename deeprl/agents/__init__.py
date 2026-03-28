"""
Module des agents.

Ce module contient les différents agents d'apprentissage par renforcement.

Catégories:
- Tabular: Q-Learning classique (table)
"""

from deeprl.agents.base import Agent
from deeprl.agents.random_agent import RandomAgent
from deeprl.agents.human_agent import HumanAgent
from deeprl.agents.tabular.q_learning import TabularQLearning

__all__ = [
    # Base
    "Agent",
    "RandomAgent",
    "HumanAgent",
    # Tabular
    "TabularQLearning",
]
