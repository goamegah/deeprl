"""
Agents basés sur la valeur (Value-Based).

Ces agents apprennent une fonction Q(s, a) et choisissent
l'action qui maximise cette valeur.
"""

from deeprl.agents.value_based.dqn import DQNAgent

__all__ = [
    "DQNAgent",
]
