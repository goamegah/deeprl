"""
Module des reseaux de neurones.

Architectures reutilisables pour les agents de deep RL :
- MLP : reseau fully-connected (DQN, REINFORCE, PPO, etc.)
"""

from deeprl.networks.mlp import MLP

__all__ = ["MLP"]
