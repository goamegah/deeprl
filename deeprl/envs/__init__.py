"""
Module des environnements de jeu.

Ce module contient les différents environnements sur lesquels
les agents peuvent s'entraîner et être évalués.

Environnements disponibles:
- LineWorld: Environnement 1D simple (tutoriel)
- GridWorld: Grille 2D avec états terminaux (goal/fail)
- TicTacToe: Morpion (jeu à 2 joueurs)
- Quarto: Jeu de stratégie avancé (2 joueurs)
"""

from deeprl.envs.base import Environment
from deeprl.envs.line_world import LineWorld
from deeprl.envs.grid_world import GridWorld
from deeprl.envs.tictactoe import TicTacToe, TicTacToeVsRandom
from deeprl.envs.quarto import Quarto, QuartoVsRandom

__all__ = [
    "Environment",
    "LineWorld",
    "GridWorld",
    "TicTacToe",
    "TicTacToeVsRandom",
    "Quarto",
    "QuartoVsRandom",
]
