"""
Module des agents.

Ce module contient les différents agents d'apprentissage par renforcement.

Catégories:
- Tabular: Q-Learning classique (table)
- Value-based: DQN et variantes (réseau de neurones)
- Policy-gradient: REINFORCE et variantes (gradient de politique)
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
from deeprl.agents.policy_gradient.reinforce import (
    REINFORCE,
    REINFORCEWithMeanBaseline,
    REINFORCEWithCriticBaseline,
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
    # Policy Gradient
    "REINFORCE",
    "REINFORCEWithMeanBaseline",
    "REINFORCEWithCriticBaseline",
    "PPO",
]
