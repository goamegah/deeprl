"""
RandomRollout — Monte Carlo sans arbre (pure rollout).

Strategie : pour chaque action disponible, simuler n_simulations parties
aleatoires depuis l'etat courant, et retourner l'action avec le meilleur
retour moyen.

Complexite : O(n_actions * n_simulations * max_depth)

Avantage par rapport a RandomAgent :
- Utilise l'information de l'environnement (simulation) pour discriminer
  les actions
- Peut detecter des actions immediatement gagnantes/perdantes

Inconvenients :
- Cout en temps proportionnel au budget de simulation
- Pas d'exploration au sens RL (pas d'apprentissage entre les coups)
- Performances limitees sur des horizons longs (rollouts biaises)

References :
- Sutton & Barto (2018), Ch. 8.1 (Models and Planning)
- Coulom (2006) "Efficient Selectivity and Backup Operators in
  Monte-Carlo Tree Search"
"""

import numpy as np
from typing import Optional, List

from deeprl.agents.base import Agent


class RandomRollout(Agent):
    """
    Planification par rollouts aleatoires (Monte Carlo without tree).

    Pour chaque action disponible, simule n_simulations parties aleatoires
    et retourne l'action avec le meilleur retour moyen estime.

    Pas d'apprentissage : la « memoire » du joueur est remise a zero a chaque
    coup. La qualite depend uniquement du budget de simulation.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_simulations: int = 50,
        max_depth: int = 200,
        gamma: float = 1.0,
    ):
        """
        Args:
            state_dim:     Dimension du vecteur d'etat (pour conformite d'interface)
            n_actions:     Nombre d'actions possibles
            n_simulations: Nombre de rollouts par action candidate
            max_depth:     Profondeur maximale d'un rollout
            gamma:         Facteur de discount (1.0 = non-discounted)
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="RandomRollout"
        )
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs,
    ) -> int:
        """
        Choisit l'action avec le meilleur retour moyen sur n_simulations
        rollouts aleatoires.

        Si l'environnement n'est pas disponible (env=None dans kwargs),
        retourne une action aleatoire.
        """
        env = kwargs.get("env")

        if available_actions is None:
            available_actions = list(range(self.n_actions))
        if not available_actions:
            return 0
        if env is None or len(available_actions) == 1:
            return int(np.random.choice(available_actions))

        best_action = available_actions[0]
        best_value = float("-inf")

        # determinize(state) reconstruit l'env depuis l'observation de l'agent
        # (plus correct que clone() qui copie l'etat interne de l'objet env)
        for action in available_actions:
            total = sum(
                self._rollout(env.determinize(state), action)
                for _ in range(self.n_simulations)
            )
            avg = total / self.n_simulations
            if avg > best_value:
                best_value = avg
                best_action = action

        return best_action

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _rollout(self, sim, first_action: int) -> float:
        """
        Simule une partie aleatoire en appliquant first_action en premier.

        Args:
            sim: Environnement deja determinize depuis l'observation courante.

        Returns:
            Retour disconte cumule du rollout.
        """
        _, reward, done = sim.step(first_action)

        total = float(reward)
        discount = self.gamma
        depth = 1

        while not done and depth < self.max_depth:
            available = sim.get_available_actions()
            if not available:
                break
            action = int(np.random.choice(available))
            _, r, done = sim.step(action)
            total += discount * float(r)
            discount *= self.gamma
            depth += 1

        return total

    # ------------------------------------------------------------------
    # Metadonnees
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "max_depth": self.max_depth,
            "gamma": self.gamma,
        }
