"""
Module des agents.

Ce module contient les différents agents d'apprentissage par renforcement.

Catégories:
- Tabular: Q-Learning classique (table)
- Value-Based: DQN et variantes (réseau de neurones)
- Policy-Based: REINFORCE, PPO (gradient de politique)
- Planning: MCTS, AlphaZero, MuZero
- Imitation: Behavior Cloning, DAgger
"""

from deeprl.agents.base import Agent
from deeprl.agents.random_agent import RandomAgent
from deeprl.agents.human_agent import HumanAgent
from deeprl.agents.tabular.q_learning import TabularQLearning
from deeprl.agents.value_based.dqn import DQNAgent
from deeprl.agents.policy_based.reinforce import REINFORCEAgent
from deeprl.agents.policy_based.ppo import PPOAgent
from deeprl.agents.planning.mcts import MCTSAgent, RandomRolloutAgent
from deeprl.agents.planning.alphazero import AlphaZeroAgent
from deeprl.agents.planning.muzero import MuZeroAgent
from deeprl.agents.planning.muzero_stochastic import StochasticMuZeroAgent
from deeprl.agents.imitation.expert_apprentice import (
    ExpertApprenticeAgent, MCTSExpert, BehaviorCloning, DAgger
)

__all__ = [
    # Base
    "Agent",
    "RandomAgent",
    "HumanAgent",
    # Tabular
    "TabularQLearning",
    # Value-Based
    "DQNAgent",
    # Policy-Based
    "REINFORCEAgent",
    "PPOAgent",
    # Planning
    "MCTSAgent",
    "RandomRolloutAgent",
    "AlphaZeroAgent",
    "MuZeroAgent",
    "StochasticMuZeroAgent",
    # Imitation
    "ExpertApprenticeAgent",
    "MCTSExpert",
    "BehaviorCloning",
    "DAgger",
]
