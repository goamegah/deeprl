"""
Agents tabulaires.

Ces agents utilisent des tables (matrices) pour stocker les valeurs Q.
Ils fonctionnent bien sur des espaces d'états discrets et petits.
"""

from deeprl.agents.tabular.q_learning import TabularQLearning

__all__ = [
    "TabularQLearning",
]
