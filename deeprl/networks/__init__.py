"""
Module des réseaux de neurones.

Ce module contient les architectures de réseaux utilisées par les agents deep RL.
"""

from deeprl.networks.mlp import MLP, DuelingMLP, ActorCriticMLP

__all__ = [
    "MLP",
    "DuelingMLP",
    "ActorCriticMLP",
]
