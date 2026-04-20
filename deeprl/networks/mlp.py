"""
Multi-Layer Perceptron (MLP) - Reseau fully-connected.

Architecture de base pour l'approximation de fonctions en RL :
- Q(s, a) pour DQN et ses variantes
- pi(a|s) pour REINFORCE et PPO
- V(s) pour les baselines et critiques

L'architecture est configurable : nombre et taille des couches cachees,
fonction d'activation, activation de sortie optionnelle.

Reference : Sutton & Barto (2018), Ch. 9.7 — Nonlinear Function Approximation
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type


class MLP(nn.Module):
    """
    Reseau de neurones fully-connected (Multi-Layer Perceptron).

    Architecture :
        input_dim -> [Linear -> Activation] x N -> Linear -> output_dim

    Exemple :
        >>> net = MLP(input_dim=27, output_dim=9, hidden_dims=[128, 64])
        >>> x = torch.randn(1, 27)
        >>> q_values = net(x)  # shape: (1, 9)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = None,
    ):
        """
        Args:
            input_dim: Dimension de l'entree (ex: state_dim)
            output_dim: Dimension de la sortie (ex: n_actions)
            hidden_dims: Tailles des couches cachees (defaut: [64, 64])
            activation: Fonction d'activation entre les couches
            output_activation: Activation optionnelle en sortie (ex: nn.Softmax)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

        # Initialisation Xavier uniform (bon defaut pour ReLU)
        self._init_weights()

    def _init_weights(self):
        """Initialise les poids avec Xavier uniform, biais a zero."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor d'entree, shape (batch, input_dim)

        Returns:
            Tensor de sortie, shape (batch, output_dim)
        """
        return self.network(x)
