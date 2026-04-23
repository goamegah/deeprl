"""
Module des agents.

Ce module contient les différents agents d'apprentissage par renforcement.

Catégories:
- Tabular: Q-Learning classique (table)
- Value-based: DQN et variantes (réseau de neurones)
- Policy-based: REINFORCE et variantes (gradient de politique)
"""

from deeprl.agents.base import Agent
from deeprl.agents.random_agent import RandomAgent
from deeprl.agents.human_agent import HumanAgent
from deeprl.agents.tabular.q_learning import TabularQLearning
from deeprl.agents.value_based.dqn import (
    DeepQLearning,
    DoubleDeepQLearning,
    DDQNWithExperienceReplay,
    DDQNWithPrioritizedExperienceReplay,
)
from deeprl.agents.policy_based.policy_gradient import (
    REINFORCE,
    REINFORCEWithBaseline,
    REINFORCEWithCritic,
    PPO,
)

__all__ = [
    # Base
    "Agent",
    "RandomAgent",
    "HumanAgent",
    # Tabular
    "TabularQLearning",
    # Value-based (DQN)
    "DeepQLearning",
    "DoubleDeepQLearning",
    "DDQNWithExperienceReplay",
    "DDQNWithPrioritizedExperienceReplay",
    # Policy-based (REINFORCE / PPO)
    "REINFORCE",
    "REINFORCEWithBaseline",
    "REINFORCEWithCritic",
    "PPO",
]
