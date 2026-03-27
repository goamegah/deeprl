"""
Agents basés sur la politique (Policy-Based).

Ces agents apprennent directement une politique π(a|s) au lieu
d'une fonction de valeur Q(s, a).

Avantages par rapport aux méthodes basées sur la valeur:
- Peuvent apprendre des politiques stochastiques
- Plus naturels pour les espaces d'actions continus
- Convergence plus stable (gradient ascent sur la performance)

Inconvénients:
- Variance élevée des gradients
- Peuvent converger vers des optima locaux
- Nécessitent souvent plus d'échantillons
"""

from deeprl.agents.policy_based.reinforce import REINFORCEAgent
from deeprl.agents.policy_based.ppo import PPOAgent

__all__ = [
    "REINFORCEAgent",
    "PPOAgent",
]
