"""
Agents de planification (Planning).

Ces agents utilisent un modele de l'environnement pour
planifier leurs actions en simulant les consequences.

Algorithmes:
- MCTS: Monte Carlo Tree Search (rollouts aleatoires)
- AlphaZero: MCTS + reseau de neurones (policy + value)
- MuZero: Modele appris de l'environnement
- MuZero Stochastique: MuZero pour environnements stochastiques
"""

from deeprl.agents.planning.mcts import MCTSAgent, RandomRolloutAgent
from deeprl.agents.planning.alphazero import AlphaZeroAgent, AlphaZeroNetwork
from deeprl.agents.planning.muzero import MuZeroAgent
from deeprl.agents.planning.muzero_stochastic import StochasticMuZeroAgent

__all__ = [
    "MCTSAgent",
    "RandomRolloutAgent",
    "AlphaZeroAgent",
    "AlphaZeroNetwork",
    "MuZeroAgent",
    "StochasticMuZeroAgent",
]
