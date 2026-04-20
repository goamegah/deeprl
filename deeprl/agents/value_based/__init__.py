"""
Agents value-based (bases sur la valeur).

Ces agents apprennent une fonction de valeur Q(s,a) pour prendre des decisions.

Progression pedagogique :
1. DeepQLearning          - reseau unique, mises a jour en ligne
2. DoubleDeepQLearning    - ajoute un reseau cible
3. DDQNWithExperienceReplay     - ajoute un buffer de rejeu
4. DDQNWithPrioritizedExperienceReplay - ajoute la priorite
"""

from deeprl.agents.value_based.dqn import (
    DeepQLearning,
    DoubleDeepQLearning,
    DDQNWithExperienceReplay,
    DDQNWithPrioritizedExperienceReplay,
)

__all__ = [
    "DeepQLearning",
    "DoubleDeepQLearning",
    "DDQNWithExperienceReplay",
    "DDQNWithPrioritizedExperienceReplay",
]
