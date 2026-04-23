"""
Monte Carlo Tree Search avec formule UCT (Upper Confidence Trees).

Algorithme en 4 phases :
  1. Selection     — descend via UCB1 jusqu'a une feuille non-expandee
  2. Expansion     — cree les enfants de la feuille (toutes les actions)
  3. Simulation    — rollout aleatoire depuis la feuille
  4. Backpropagation — remonte la valeur dans l'arbre

Formule UCB1 (bandit multi-bras) :
    UCB1(s, a) = Q(s,a) + c * sqrt( ln(N(s) + 1) / N(s,a) )
ou :
  Q(s,a) = W(s,a) / N(s,a)  — valeur moyenne des simulations
  N(s)   — visites du noeud parent
  N(s,a) — visites de l'enfant
  c      — coefficient d'exploration (sqrt(2) ≈ 1.41 est la valeur theorique)

Avantage sur RandomRollout :
- Concentre les simulations sur les regions prometteuses de l'arbre
- Convergence theorique vers l'action optimale (theoreme de Kocsis)
- Pas besoin de fonctions d'evaluation

MCTSNode est exporte car AlphaZero le reutilise.

References :
- Kocsis & Szepesvari (2006) "Bandit based Monte-Carlo Planning"
- Browne et al. (2012) "A Survey of Monte Carlo Tree Search Methods"
- Sutton & Barto (2018), Ch. 8.11 (Monte Carlo Tree Search)
"""

import math
import random

import numpy as np
from typing import Optional, List, Dict

from deeprl.agents.base import Agent


# ============================================================================
# Noeud MCTS (partage avec AlphaZero)
# ============================================================================

class MCTSNode:
    """
    Noeud de l'arbre MCTS.

    Attributs :
        N      : nombre de visites
        W      : somme des valeurs reçues (backprop)
        prior  : probabilite a priori P(s,a), utilisee par PUCT (AlphaZero)
        children : dictionnaire { action -> MCTSNode }

    Les slots accelerent l'acces et reduisent la memoire (nombreux noeuds).
    """

    __slots__ = ("N", "W", "prior", "children")

    def __init__(self, prior: float = 1.0):
        self.N: int = 0
        self.W: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, "MCTSNode"] = {}

    # ------------------------------------------------------------------
    # Valeur estimee
    # ------------------------------------------------------------------

    @property
    def Q(self) -> float:
        """Valeur moyenne des simulations passant par ce noeud."""
        return self.W / self.N if self.N > 0 else 0.0

    # ------------------------------------------------------------------
    # Formules de selection
    # ------------------------------------------------------------------

    def ucb1(self, parent_N: int, c: float) -> float:
        """
        UCB1 — Upper Confidence Bound (Auer et al., 2002).

        Nodes non-visites ont score +inf (priorite absolue a la premiere visite).
        """
        if self.N == 0:
            return float("inf")
        return self.Q + c * math.sqrt(math.log(parent_N + 1) / self.N)

    def puct(self, parent_N: int, c: float) -> float:
        """
        PUCT — Polynomial Upper Confidence bound for Trees (AlphaZero).

        Integre le prior reseau P(s,a) pour guider l'exploration.
            PUCT(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        return self.Q + c * self.prior * math.sqrt(parent_N) / (1 + self.N)

    # ------------------------------------------------------------------
    # Mise a jour
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """Incremente N et accumule la valeur (backpropagation)."""
        self.N += 1
        self.W += value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_expanded(self) -> bool:
        return bool(self.children)

    def best_action_by_visits(self) -> int:
        """Retourne l'action enfant la plus visitee (politique finale)."""
        return max(self.children, key=lambda a: self.children[a].N)

    def visit_counts_as_policy(
        self, n_actions: int, temperature: float = 1.0
    ) -> np.ndarray:
        """
        Convertit les comptes de visites en distribution de politique.

        temperature = 0  → argmax (deterministe)
        temperature = 1  → proportionnel aux visites
        temperature → ∞  → uniforme
        """
        counts = np.zeros(n_actions, dtype=np.float32)
        for a, child in self.children.items():
            counts[a] = float(child.N)

        if temperature == 0.0 or counts.sum() == 0:
            probs = np.zeros(n_actions, dtype=np.float32)
            if counts.sum() > 0:
                probs[int(np.argmax(counts))] = 1.0
            return probs

        powered = counts ** (1.0 / temperature)
        return powered / powered.sum()


# ============================================================================
# Agent MCTS (UCT)
# ============================================================================

class MCTS(Agent):
    """
    Monte Carlo Tree Search (UCT).

    Planification pure — aucun apprentissage de parametres. Construit un
    arbre de recherche a chaque coup en simulant n_simulations parties
    aleatoires depuis l'etat courant.

    La force de MCTS vient de la concentration progressive des simulations
    sur les branches les plus prometteuses (trade-off exploration/exploitation
    garanti par UCB1).

    Pas de parametre appris : la qualite depend uniquement du budget
    de simulation (n_simulations).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_simulations: int = 100,
        c_puct: float = 1.41,
        max_depth: int = 200,
        gamma: float = 1.0,
    ):
        """
        Args:
            state_dim:     Dimension du vecteur d'etat
            n_actions:     Nombre d'actions
            n_simulations: Budget de simulation par coup
            c_puct:        Coefficient d'exploration UCB1 (sqrt(2) ≈ 1.41)
            max_depth:     Profondeur maximale des rollouts
            gamma:         Facteur de discount dans les rollouts
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="MCTS"
        )
        self.n_simulations = n_simulations
        self.c_puct = c_puct
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
        Retourne l'action avec le plus grand nombre de visites dans l'arbre.

        Si l'environnement n'est pas disponible, retourne une action aleatoire.
        """
        env = kwargs.get("env")

        if available_actions is None:
            available_actions = list(range(self.n_actions))
        if not available_actions:
            return 0
        if env is None or len(available_actions) == 1:
            return int(np.random.choice(available_actions))

        # Initialiser la racine avec les actions disponibles
        root = MCTSNode()
        for a in available_actions:
            root.children[a] = MCTSNode()

        # Executer n_simulations iterations
        # determinize(state) reconstruit l'env depuis l'observation de l'agent
        for _ in range(self.n_simulations):
            sim = env.determinize(state)
            self._simulate(root, sim)

        return root.best_action_by_visits()

    # ------------------------------------------------------------------
    # Simulation MCTS (4 phases)
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode, sim) -> float:
        """
        Une simulation MCTS complete :
        1. Selection : UCB1 jusqu'a une feuille
        2. Expansion : cree les enfants de la feuille
        3. Rollout   : partie aleatoire depuis la feuille
        4. Backprop  : met a jour tous les noeuds du chemin
        """
        node = root
        path: List[MCTSNode] = [node]
        terminal_reward = 0.0

        # --- 1 & 2. Selection + Expansion ---
        while not sim.is_game_over:
            if not node.is_expanded():
                # Expansion : cree tous les enfants disponibles
                available = sim.get_available_actions()
                if not available:
                    break
                for a in available:
                    node.children[a] = MCTSNode()

                # Visite un enfant non visite (expansion selective)
                unvisited = [
                    a for a, c in node.children.items() if c.N == 0
                ]
                action = random.choice(unvisited)
                _, r, _ = sim.step(action)
                terminal_reward = float(r)  # capture la recompense (terminale ou non)
                node = node.children[action]
                path.append(node)
                break  # Arret apres expansion
            else:
                # Selection : meilleur enfant par UCB1
                action = max(
                    node.children,
                    key=lambda a: node.children[a].ucb1(node.N, self.c_puct),
                )
                _, r, _ = sim.step(action)
                terminal_reward = float(r)
                node = node.children[action]
                path.append(node)

        # --- 3. Simulation (rollout aleatoire) ---
        # Si le jeu est fini, on utilise la recompense terminale capturee.
        # Sinon, on continue avec un rollout aleatoire.
        if sim.is_game_over:
            value = terminal_reward
        else:
            value = self._random_rollout(sim)

        # --- 4. Backpropagation ---
        for n in reversed(path):
            n.update(value)

        return value

    def _random_rollout(self, sim) -> float:
        """Rollout aleatoire depuis l'etat courant de sim."""
        total = 0.0
        discount = 1.0
        depth = 0

        while not sim.is_game_over and depth < self.max_depth:
            available = sim.get_available_actions()
            if not available:
                break
            action = int(np.random.choice(available))
            _, r, _ = sim.step(action)
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
            "c_puct": self.c_puct,
            "max_depth": self.max_depth,
            "gamma": self.gamma,
        }
