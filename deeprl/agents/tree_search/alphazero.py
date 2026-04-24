"""
AlphaZero — MCTS guide par un reseau politique + valeur.

Differences cles avec MCTS standard :
  1. Selection via PUCT (utilise les priors P(s,a) du reseau)
  2. Evaluation : le reseau estime V(s) — pas de rollout aleatoire
  3. Apprentissage : le reseau est entraine sur (state, mcts_policy, G_t)

Architecture du reseau :
  - Tronc commun (MLP)
  - Tete politique : logits des actions (cross-entropie vs politiques MCTS)
  - Tete valeur    : scalaire V(s) ∈ [-1,1] (MSE vs retours discountes)

Boucle d'entrainement :
  - act()   : MCTS guide par le reseau → retourne l'action + stocke la politique
  - learn() : accumule (state, mcts_policy, reward) ; en fin d'episode :
              calcule G_t, alimente le replay, entraine le reseau

En evaluation (training=False) :
  - Le reseau seul est utilise (sans MCTS), pour la rapidite.

Formule PUCT (Silver et al., 2017) :
  PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

References :
  Silver et al. (2017) "Mastering Chess and Shogi by Self-Play with a
    General Reinforcement Learning Algorithm" (AlphaZero)
  Silver et al. (2016) "Mastering the game of Go with deep neural networks
    and tree search" (AlphaGo)
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprl.agents.base import Agent
from deeprl.agents.tree_search.mcts import MCTSNode


# ============================================================================
# Reseau politique + valeur (shared trunk)
# ============================================================================

class PolicyValueNetwork(nn.Module):
    """
    Reseau partage pour AlphaZero.

    Tronc commun suivi de deux tetes :
      - policy_head : logits (n_actions)  — cible = distribution MCTS
      - value_head  : scalaire tanh ∈ [-1,1] — cible = retour disconte
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: List[int],
    ):
        super().__init__()

        # Tronc
        trunk_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Tete politique
        self.policy_head = nn.Linear(prev, n_actions)

        # Tete valeur
        mid = max(prev // 2, 16)
        self.value_head = nn.Sequential(
            nn.Linear(prev, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            policy_logits : (batch, n_actions)
            value         : (batch, 1)  ∈ [-1, 1]
        """
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)


# ============================================================================
# Agent AlphaZero
# ============================================================================

class AlphaZero(Agent):
    """
    AlphaZero — MCTS guide par un reseau neuronal.

    Amelioration par rapport a MCTS :
      - La selection utilise les priors du reseau (PUCT) → moins de simulations
        necessaires pour converger vers une bonne action
      - La feuille est evaluee par V(s) (pas de rollout aleatoire) → valeurs
        plus precises des que le reseau est un peu entraine

    Cycle complet d'un episode :
      act()   → MCTS PUCT → politique de visite → action echantillonnee
      learn() → accumule transitions → en fin d'episode : entraine le reseau
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        n_simulations: int = 50,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        min_buffer_size: int = 500,
        l2_reg: float = 1e-4,
    ):
        """
        Args:
            state_dim:       Dimension du vecteur d'etat
            n_actions:       Nombre d'actions
            hidden_dims:     Couches cachees du MLP ([128, 128] par defaut)
            lr:              Learning rate (Adam)
            gamma:           Facteur de discount pour les retours G_t
            n_simulations:   Budget MCTS par coup (en entrainement)
            c_puct:          Coefficient d'exploration PUCT
            temperature:     Temperature pour l'echantillonnage de la politique
                             (0 = argmax, 1 = proportionnel aux visites)
            buffer_size:     Taille du replay buffer
            batch_size:      Taille de batch pour l'optimisation
            min_buffer_size: Taille minimale avant de commencer l'apprentissage
            l2_reg:          Regularisation L2 (weight decay)
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="AlphaZero"
        )

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.gamma = gamma
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.l2_reg = l2_reg

        # Reseau
        self.network = PolicyValueNetwork(
            state_dim, n_actions, hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=l2_reg
        )

        # Replay buffer : tuples (state, mcts_policy, return)
        self._replay: deque = deque(maxlen=buffer_size)

        # Buffers episode
        self._ep_states: List[np.ndarray] = []
        self._ep_policies: List[np.ndarray] = []
        self._ep_rewards: List[float] = []

        # Bridge act → learn
        self._last_mcts_policy: Optional[np.ndarray] = None
        self._training_mode: bool = True

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
        Choisit une action via MCTS (en entrainement) ou via le reseau seul
        (en evaluation) pour eviter le cout du MCTS a l'inference.
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        if not available_actions:
            return 0

        env = kwargs.get("env")

        if training and self._training_mode and env is not None:
            mcts_policy = self._run_mcts(state, available_actions, env)
        else:
            mcts_policy = self._network_policy_masked(state, available_actions)

        self._last_mcts_policy = mcts_policy

        temp = self.temperature if (training and self._training_mode) else 0.0
        if temp > 0.0:
            action = int(np.random.choice(self.n_actions, p=mcts_policy))
        else:
            action = int(np.argmax(mcts_policy))

        return action

    # ------------------------------------------------------------------
    # MCTS avec PUCT
    # ------------------------------------------------------------------

    def _run_mcts(
        self,
        state: np.ndarray,
        available_actions: List[int],
        env,
    ) -> np.ndarray:
        """
        Execute n_simulations de MCTS avec PUCT et retourne la distribution
        de visite normalisee (politique MCTS).
        """
        root = MCTSNode()

        # Initialiser la racine avec les priors du reseau
        priors, _ = self._network_output(state)
        for a in available_actions:
            root.children[a] = MCTSNode(prior=float(priors[a]))

        # determinize(state) reconstruit l'env depuis l'observation de l'agent
        for _ in range(self.n_simulations):
            sim = env.determinize(state)
            self._mcts_simulate(root, sim)

        return root.visit_counts_as_policy(self.n_actions, self.temperature)

    def _mcts_simulate(self, root: MCTSNode, sim) -> float:
        """
        Une simulation MCTS avec PUCT + evaluation reseau :
          - Selection  : PUCT jusqu'a feuille
          - Expansion  : priors du reseau pour les enfants
          - Evaluation : V(s) du reseau (pas de rollout aleatoire)
          - Backprop   : remonte la valeur
        """
        node = root
        path: List[MCTSNode] = [node]
        terminal_reward = 0.0

        # Selection via PUCT
        while node.is_expanded() and not sim.is_game_over:
            action = max(
                node.children,
                key=lambda a: node.children[a].puct(node.N, self.c_puct),
            )
            _, r, _ = sim.step(action)
            terminal_reward = float(r)
            node = node.children[action]
            path.append(node)

        # Evaluation + Expansion
        if sim.is_game_over:
            value = terminal_reward  # recompense terminale capturee lors du dernier step
        else:
            leaf_state = sim.get_state()
            priors, value = self._network_output(leaf_state)

            # Expansion avec priors du reseau
            available = sim.get_available_actions()
            if available:
                for a in available:
                    node.children[a] = MCTSNode(prior=float(priors[a]))

        # Backpropagation
        for n in reversed(path):
            n.update(value)

        return value

    # ------------------------------------------------------------------
    # Appels reseau
    # ------------------------------------------------------------------

    def _network_output(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Appel reseau → (policy_probs numpy, value float). Sans gradient."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value_t = self.network(state_t)
        policy = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return policy, float(value_t.item())

    def _network_policy_masked(
        self, state: np.ndarray, available_actions: List[int]
    ) -> np.ndarray:
        """Politique masquee via le reseau seul (evaluation sans MCTS)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.network(state_t)

        mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
        for a in available_actions:
            mask[a] = 0.0
        masked = logits.squeeze(0) + mask
        return F.softmax(masked, dim=-1).cpu().numpy()

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
        """
        Collecte la transition. Entraine le reseau en fin d'episode.
        """
        if self._last_mcts_policy is not None:
            self._ep_states.append(state.copy())
            self._ep_policies.append(self._last_mcts_policy.copy())
            self._ep_rewards.append(float(reward))
            self._last_mcts_policy = None

        if done and len(self._ep_states) > 0:
            return self._finish_episode()

        return None

    def _finish_episode(self) -> Dict[str, float]:
        """Calcule retours, alimente replay, entraine le reseau."""
        returns = self._compute_returns()

        for s, p, g in zip(self._ep_states, self._ep_policies, returns):
            self._replay.append((s.copy(), p.copy(), float(g)))

        self._ep_states.clear()
        self._ep_policies.clear()
        self._ep_rewards.clear()

        if len(self._replay) < self.min_buffer_size:
            return {"replay_size": float(len(self._replay))}

        return self._train_batch()

    def _compute_returns(self) -> List[float]:
        """Retours discountes G_t = r_t + gamma * r_{t+1} + ..."""
        T = len(self._ep_rewards)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = self._ep_rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def _train_batch(self) -> Dict[str, float]:
        """Echantillonne un batch et met a jour le reseau."""
        n = min(self.batch_size, len(self._replay))
        batch = random.sample(self._replay, n)
        states, policies, returns = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        policies_t = torch.FloatTensor(np.array(policies)).to(self.device)
        returns_t = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        logits, values = self.network(states_t)

        # Politique : cross-entropie entre politique MCTS et distribution predite
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(policies_t * log_probs).sum(dim=-1).mean()

        # Valeur : MSE entre retours discountes et V(s) predit
        value_loss = F.mse_loss(values, returns_t)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def on_episode_start(self) -> None:
        self._ep_states.clear()
        self._ep_policies.clear()
        self._ep_rewards.clear()
        self._last_mcts_policy = None

    def set_training_mode(self, training: bool) -> None:
        self._training_mode = training
        if training:
            self.network.train()
        else:
            self.network.eval()

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "class": "AlphaZero",
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "n_simulations": self.n_simulations,
                "c_puct": self.c_puct,
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)

    def get_config(self) -> Dict:
        return {
            "n_simulations": self.n_simulations,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "lr": self.lr,
            "gamma": self.gamma,
            "hidden_dims": self.hidden_dims,
        }
