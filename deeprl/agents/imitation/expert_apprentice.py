"""
Expert Apprentice — Apprentissage par imitation de MCTS.

Principe (Behavioral Cloning depuis un expert MCTS) :
  1. L'expert (MCTS) fournit la « bonne » politique au step courant
  2. L'eleve (reseau neuronal) apprend a imiter cette politique par
     regression supervisee (cross-entropie)
  3. A l'inference, l'eleve seul est utilise (pas de MCTS → inference rapide)

Differences avec AlphaZero :
  - Pas d'auto-amelioration : l'expert MCTS est fixe, pas guide par le reseau
  - Entrainement purement supervisé (pas de self-play)
  - L'eleve n'influence pas les donnees de training (pas de DAgger)
  - Plus simple a entrainer ; moins performant sur l'horizon long

Differences avec MCTS :
  - A l'inference : reseau seul (O(1) par coup) vs MCTS (O(n_simulations) par coup)
  - Generalisation : le reseau interpole entre les etats vus pendant l'imitation

Lien avec AlphaZero :
  ExpertApprentice ≈ AlphaZero iteration 0 (avant que le reseau devienne
  l'expert lui-meme). C'est le sous-probleme de « distillation MCTS → reseau ».

Variante DAgger (non implémentée ici) :
  Pour aller plus loin, DAgger ameliore BC en recollectant des demonstrations
  expert sur les etats VISITES par la politique apprise (correction de
  la distribution shift). Voir Ross & Bagnell (2011).

References :
  Pomerleau (1989) "ALVINN: An Autonomous Land Vehicle In a Neural Network"
    (Behavioral Cloning original)
  Ross et al. (2011) "A Reduction of Imitation Learning and Structured
    Prediction to No-Regret Online Learning" (DAgger)
  Silver et al. (2017) AlphaZero (distillation MCTS → reseau)
"""

import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprl.agents.base import Agent
from deeprl.agents.tree_search.mcts import MCTS, MCTSNode
from deeprl.networks.mlp import MLP


class ExpertApprentice(Agent):
    """
    Expert Apprentice (Behavioral Cloning depuis MCTS).

    Deux phases entrelacees a chaque episode :
      act()   — collecte la politique expert MCTS et l'action correspondante
      learn() — accumule les paires (state, expert_policy) et, en fin
                d'episode, entraine l'eleve par cross-entropie

    Modes :
      training=True  → l'expert MCTS choisit l'action (demonstrations)
      training=False → le reseau eleve choisit l'action (evaluation)

    Parametre `use_student_ratio` (entre 0 et 1) :
      Fraction des episodes ou l'eleve est utilise a la place de l'expert
      (protocole DAgger simplifie, permet de corriger la distribution shift).
      0.0 = BC pur (expert toujours), 1.0 = eleve toujours.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        # Paramètres MCTS expert
        n_simulations: int = 50,
        c_puct: float = 1.41,
        max_depth: int = 200,
        gamma: float = 1.0,
        # Paramètres d'apprentissage
        buffer_size: int = 20_000,
        batch_size: int = 64,
        min_buffer_size: int = 500,
        l2_reg: float = 1e-4,
        # DAgger simplifie
        use_student_ratio: float = 0.0,
    ):
        """
        Args:
            state_dim:        Dimension du vecteur d'etat
            n_actions:        Nombre d'actions
            hidden_dims:      Architecture du reseau eleve
            lr:               Learning rate
            n_simulations:    Budget MCTS de l'expert par coup
            c_puct:           Coefficient d'exploration UCB1 de l'expert
            max_depth:        Profondeur maximale des rollouts expert
            gamma:            Discount des rollouts expert
            buffer_size:      Taille du replay buffer (state, expert_policy)
            batch_size:       Taille de batch
            min_buffer_size:  Debut de l'apprentissage
            l2_reg:           Regularisation L2
            use_student_ratio: Fraction d'episodes ou l'eleve collecte les
                               donnees (DAgger simplifie, 0 = BC pur)
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="ExpertApprentice"
        )

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.lr = lr
        self.hidden_dims = hidden_dims
        self.use_student_ratio = use_student_ratio
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.l2_reg = l2_reg
        self.gamma = gamma

        # Expert MCTS (pas d'apprentissage)
        self._expert = MCTS(
            state_dim=state_dim,
            n_actions=n_actions,
            n_simulations=n_simulations,
            c_puct=c_puct,
            max_depth=max_depth,
            gamma=gamma,
        )

        # Reseau eleve : state → policy logits
        self.student_net = MLP(state_dim, n_actions, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.student_net.parameters(), lr=lr, weight_decay=l2_reg
        )

        # Replay : paires (state, expert_policy)
        self._replay: deque = deque(maxlen=buffer_size)

        # Buffers episode
        self._ep_states: List[np.ndarray] = []
        self._ep_expert_policies: List[np.ndarray] = []

        # Pont act → learn
        self._last_expert_policy: Optional[np.ndarray] = None
        self._training_mode: bool = True

        # Pour DAgger : decider si l'episode courant utilise l'eleve
        self._use_student_this_episode: bool = False

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
        Retourne l'action de l'expert (en entrainement) ou de l'eleve
        (en evaluation ou si use_student_this_episode).
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        if not available_actions:
            return 0

        env = kwargs.get("env")

        # --- Expert MCTS ---
        if env is not None:
            expert_policy = self._expert_mcts_policy(
                state, available_actions, env
            )
        else:
            # Pas d'env disponible : politique uniforme sur les actions valides
            expert_policy = np.zeros(self.n_actions, dtype=np.float32)
            for a in available_actions:
                expert_policy[a] = 1.0 / len(available_actions)

        self._last_expert_policy = expert_policy

        # Choix de qui agit
        use_student = (
            not training
            or not self._training_mode
            or self._use_student_this_episode
        )

        if use_student:
            return self._student_action(state, available_actions)
        else:
            # Expert : echantillonne depuis la politique MCTS
            action = int(np.random.choice(self.n_actions, p=expert_policy))
            return action

    def _expert_mcts_policy(
        self,
        state: np.ndarray,
        available_actions: List[int],
        env,
    ) -> np.ndarray:
        """
        Lance MCTS et retourne la distribution de visite (politique expert).
        Utilise temperature=1.0 pour garder la diversite des demonstrations.
        """
        root = MCTSNode()
        for a in available_actions:
            root.children[a] = MCTSNode()

        # Memoriser le joueur que nous optimisons avant toute simulation
        our_player = env.determinize(state)._current_player
        for _ in range(self._expert.n_simulations):
            # determinize(state) reconstruit l'env depuis l'observation courante
            sim = env.determinize(state)
            self._expert._simulate(root, sim, our_player)

        return root.visit_counts_as_policy(self.n_actions, temperature=1.0)

    def _student_action(
        self, state: np.ndarray, available_actions: List[int]
    ) -> int:
        """L'eleve choisit l'action (argmax masque)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.student_net(state_t).squeeze(0)

        mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
        for a in available_actions:
            mask[a] = 0.0
        masked = logits + mask
        return int(masked.argmax().item())

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs,
    ) -> Optional[Dict[str, float]]:
        """Collecte (state, expert_policy); entraine en fin d'episode."""
        if self._last_expert_policy is not None:
            self._ep_states.append(state.copy())
            self._ep_expert_policies.append(self._last_expert_policy.copy())
            self._last_expert_policy = None

        if done and len(self._ep_states) > 0:
            return self._finish_episode()

        return None

    def _finish_episode(self) -> Dict[str, float]:
        """Alimente le replay et entraine si assez de donnees."""
        for s, p in zip(self._ep_states, self._ep_expert_policies):
            self._replay.append((s.copy(), p.copy()))

        self._ep_states.clear()
        self._ep_expert_policies.clear()

        if len(self._replay) < self.min_buffer_size:
            return {"replay_size": float(len(self._replay))}

        return self._train_batch()

    def _train_batch(self) -> Dict[str, float]:
        """Cross-entropie : eleve imite la politique expert MCTS."""
        n = min(self.batch_size, len(self._replay))
        batch = random.sample(self._replay, n)
        states, policies = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        policies_t = torch.FloatTensor(np.array(policies)).to(self.device)

        logits = self.student_net(states_t)
        log_probs = F.log_softmax(logits, dim=-1)

        # Cross-entropie : -sum_a expert_policy(a) * log student_policy(a)
        loss = -(policies_t * log_probs).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.student_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        return {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def on_episode_start(self) -> None:
        self._ep_states.clear()
        self._ep_expert_policies.clear()
        self._last_expert_policy = None

        # DAgger : decider si l'eleve est utilise cet episode
        self._use_student_this_episode = (
            random.random() < self.use_student_ratio
        )

    def set_training_mode(self, training: bool) -> None:
        self._training_mode = training
        if training:
            self.student_net.train()
        else:
            self.student_net.eval()

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "class": "ExpertApprentice",
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "student_net": self.student_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.student_net.load_state_dict(ckpt["student_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)

    def get_config(self) -> Dict:
        return {
            "expert": self._expert.get_config(),
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "use_student_ratio": self.use_student_ratio,
        }
