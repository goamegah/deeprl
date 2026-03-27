"""
Module de mémoire pour l'Experience Replay.

L'Experience Replay est une technique cruciale pour stabiliser
l'entraînement des algorithmes Deep RL comme DQN.
"""

from deeprl.memory.replay_buffer import ReplayBuffer, Transition, EpisodeBuffer
from deeprl.memory.prioritized_buffer import PrioritizedReplayBuffer

__all__ = [
    "ReplayBuffer",
    "Transition",
    "EpisodeBuffer",
    "PrioritizedReplayBuffer",
]
