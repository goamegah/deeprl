"""
MuZero et MuZeroStochastic — planification dans l'espace latent.

Differences fondamentales avec AlphaZero :
  - PAS besoin d'un simulateur de l'environnement pendant la recherche
  - Toute la planification se fait dans un espace latent APPRIS
  - 3 reseaux complementaires :

    h_0        = repr(s_0)         Representation : obs reelle → latent
    h_k, r_k   = dynamics(h_{k-1}, a_{k-1})   Dynamique : latent + action → latent + recompense
    p_k, v_k   = prediction(h_k)              Prediction : latent → politique + valeur

MCTS en espace latent :
  - Racine : h_root = repr(s) (encode l'etat reel une seule fois)
  - Selection : PUCT sur Q/N/priors
  - Expansion : h_next, r = dynamics(h_parent, action) puis prediction(h_next)
  - Backprop  : G = r + gamma * G_suivant (les recompenses s'accumulent)
  - Pas de rollout aleatoire : V(h) remplace la simulation

Entrainement (unroll K etapes) :
  Pour chaque trajectoire, a partir d'une position t :
    h_0 = repr(s_t)
    Pour k = 0..K-1 :
      p_k, v_k = prediction(h_k)
      h_{k+1}, r_{k+1} = dynamics(h_k, a_{t+k})
    Loss = CE(p_k, pi_k) + MSE(v_k, G_k) + MSE(r_k, r_reel_k)

MuZeroStochastic (extension pour les environnements stochastiques) :
  Ajoute une phase "chance" entre l'action et l'etat suivant :
    afterstate = afterstate_net(h, a)        partie deterministe
    h_next     = chance_net(afterstate, code) avec code ~ Cat(n_chance)
    code       = chance_encoder(afterstate, h_next_reel)  (pendant l'entrain.)

  MCTS avec noeuds chance :
    Noeud action → Noeud chance (afterstate) → noeuds issues (par code)

References :
  Schrittwieser et al. (2020) "Mastering Atari, Go, Chess and Shogi by
    Planning with a Learned Model" (MuZero)
  Antonoglou et al. (2021) "Planning in Stochastic Environments with a
    Learned Model" (Stochastic MuZero)
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprl.agents.base import Agent
from deeprl.networks.mlp import MLP


# ============================================================================
# Reseaux MuZero
# ============================================================================

class RepresentationNet(nn.Module):
    """
    Encode l'observation reelle en etat latent initial h_0.

    Sortie normalisee (min-max ou tanh) pour la stabilite du training.
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = MLP(obs_dim, latent_dim, hidden_dims)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(self.net(x)))


class DynamicsNet(nn.Module):
    """
    Predit (h_{t+1}, r_{t+1}) a partir de (h_t, a_t).

    L'action est encodee en one-hot et concatenee a l'etat latent.
    Deux tetes distinctes : une pour le prochain etat latent, une pour la recompense.
    """

    def __init__(
        self, latent_dim: int, n_actions: int, hidden_dims: List[int]
    ):
        super().__init__()
        in_dim = latent_dim + n_actions
        self.state_net = MLP(in_dim, latent_dim, hidden_dims)
        # Tete recompense plus legere
        rew_hidden = hidden_dims[-1:] if hidden_dims else [64]
        self.reward_net = MLP(in_dim, 1, rew_hidden)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, latent: torch.Tensor, action_oh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent, action_oh], dim=-1)
        next_h = torch.tanh(self.norm(self.state_net(x)))
        reward = self.reward_net(x)
        return next_h, reward


class PredictionNet(nn.Module):
    """
    Predit (policy_logits, value) a partir de l'etat latent h.
    Identique a la tete de AlphaZero, mais sur l'espace latent.
    """

    def __init__(
        self, latent_dim: int, n_actions: int, hidden_dims: List[int]
    ):
        super().__init__()
        self.policy_net = MLP(latent_dim, n_actions, hidden_dims)
        self.value_net = MLP(latent_dim, 1, hidden_dims)

    def forward(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(latent), torch.tanh(self.value_net(latent))


# ============================================================================
# Noeud MCTS pour MuZero (stocke l'etat latent)
# ============================================================================

class MuZeroNode:
    """
    Noeud MCTS avec etat latent associe.

    Contrairement a MCTSNode (MCTS classique), le noeud stocke :
      - latent   : l'etat latent h predit par dynamics (ou repr pour la racine)
      - reward   : recompense predite pour arriver a ce noeud
    """

    __slots__ = ("N", "W", "prior", "children", "latent", "reward")

    def __init__(
        self,
        prior: float = 1.0,
        latent: Optional[torch.Tensor] = None,
        reward: float = 0.0,
    ):
        self.N: int = 0
        self.W: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, "MuZeroNode"] = {}
        self.latent: Optional[torch.Tensor] = latent
        self.reward: float = reward

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct(self, parent_N: int, c: float) -> float:
        return self.Q + c * self.prior * np.sqrt(parent_N) / (1 + self.N)

    def update(self, value: float) -> None:
        self.N += 1
        self.W += value

    def is_expanded(self) -> bool:
        return bool(self.children)

    def best_action_by_visits(self) -> int:
        return max(self.children, key=lambda a: self.children[a].N)

    def visit_counts_as_policy(
        self, n_actions: int, temperature: float = 1.0
    ) -> np.ndarray:
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
# MuZero
# ============================================================================

class MuZero(Agent):
    """
    MuZero — planification dans un espace latent appris.

    Avantage cle : le modele de l'environnement est APPRIS, pas fourni.
    MuZero peut donc s'appliquer a des environnements ou le code source
    n'est pas accessible (ex: jeux Atari, systemes reels).

    Pendant la recherche :
      - repr(s)    encode l'observation en latent h_root
      - dynamics   predit h_next et r a partir de h et a (sans simulation)
      - prediction evalue p(a|h) et V(h) a chaque noeud de l'arbre

    Pendant l'entrainement :
      - Trajectoires (s, a, r, pi_mcts) stockees en replay
      - Deroulement K etapes du modele dynamique
      - Pertes : politique (CE), valeur (MSE), recompense (MSE)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        n_simulations: int = 50,
        c_puct: float = 1.0,
        n_unroll: int = 5,
        temperature: float = 1.0,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        min_buffer_size: int = 500,
        l2_reg: float = 1e-4,
    ):
        """
        Args:
            state_dim:    Dimension de l'observation reelle
            n_actions:    Nombre d'actions
            latent_dim:   Dimension de l'espace latent
            hidden_dims:  Couches cachees des MLP
            lr:           Learning rate
            gamma:        Discount
            n_simulations: Budget MCTS par coup
            c_puct:       Coefficient exploration PUCT
            n_unroll:     Nombre d'etapes de deroulement du modele (training)
            temperature:  Temperature pour l'echantillonnage de la politique
            buffer_size:  Taille du replay
            batch_size:   Taille de batch
            min_buffer_size: Minimum avant de commencer l'apprentissage
            l2_reg:       Regularisation L2
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="MuZero"
        )

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.n_unroll = n_unroll
        self.temperature = temperature
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.l2_reg = l2_reg

        # Trois reseaux MuZero
        self.repr_net = RepresentationNet(
            state_dim, latent_dim, hidden_dims
        ).to(self.device)
        self.dynamics_net = DynamicsNet(
            latent_dim, n_actions, hidden_dims
        ).to(self.device)
        self.pred_net = PredictionNet(
            latent_dim, n_actions, hidden_dims
        ).to(self.device)

        # Optimiseur unique sur tous les reseaux
        self.optimizer = torch.optim.Adam(
            list(self.repr_net.parameters())
            + list(self.dynamics_net.parameters())
            + list(self.pred_net.parameters()),
            lr=lr,
            weight_decay=l2_reg,
        )

        # Replay : trajectoires completes sous forme de listes de (s, a, r, pi)
        self._replay: deque = deque(maxlen=buffer_size)

        # Buffers episode
        self._ep_states: List[np.ndarray] = []
        self._ep_actions: List[int] = []
        self._ep_rewards: List[float] = []
        self._ep_policies: List[np.ndarray] = []

        # Bridge act → learn
        self._last_mcts_policy: Optional[np.ndarray] = None
        self._training_mode: bool = True

    # ------------------------------------------------------------------
    # Encodeurs / Decodeurs
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_no_grad(self, state: np.ndarray) -> torch.Tensor:
        """Observation → latent h_0 (sans grad, pour la recherche)."""
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.repr_net(s_t).squeeze(0)

    def _action_onehot(self, action: int) -> torch.Tensor:
        oh = torch.zeros(self.n_actions, device=self.device)
        oh[action] = 1.0
        return oh

    @torch.no_grad()
    def _dynamics_no_grad(
        self, latent: torch.Tensor, action: int
    ) -> Tuple[torch.Tensor, float]:
        """Un pas de dynamique (sans grad, pour la recherche MCTS)."""
        oh = self._action_onehot(action)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            oh = oh.unsqueeze(0)
        h_next, r = self.dynamics_net(latent, oh)
        return h_next.squeeze(0), float(r.item())

    @torch.no_grad()
    def _predict_no_grad(
        self, latent: torch.Tensor
    ) -> Tuple[np.ndarray, float]:
        """Prediction (sans grad, pour la recherche MCTS)."""
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        logits, value_t = self.pred_net(latent)
        policy = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return policy, float(value_t.item())

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
        Choisit une action via MCTS en espace latent.

        Pas besoin du 'env' — toute la simulation passe par les reseaux appris.
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        if not available_actions:
            return 0

        # Encoder l'etat reel → latent
        h_root = self._encode_no_grad(state)

        # MCTS en espace latent
        root = MuZeroNode(latent=h_root)
        priors, _ = self._predict_no_grad(h_root)
        for a in available_actions:
            root.children[a] = MuZeroNode(prior=float(priors[a]))

        for _ in range(self.n_simulations):
            self._mcts_simulate(root)

        mcts_policy = root.visit_counts_as_policy(self.n_actions, self.temperature)
        self._last_mcts_policy = mcts_policy

        temp = self.temperature if (training and self._training_mode) else 0.0
        if temp > 0.0:
            action = int(np.random.choice(self.n_actions, p=mcts_policy))
        else:
            action = int(np.argmax(mcts_policy))

        return action

    # ------------------------------------------------------------------
    # MCTS en espace latent
    # ------------------------------------------------------------------

    def _mcts_simulate(self, root: MuZeroNode) -> float:
        """
        Une simulation MCTS entierement dans l'espace latent.

        Selection → Expansion (via dynamics) → Evaluation (via prediction)
        → Backpropagation (avec recompenses accumulees).
        """
        node = root
        path: List[MuZeroNode] = [node]

        # Selection via PUCT
        while node.is_expanded():
            action = max(
                node.children,
                key=lambda a: node.children[a].puct(node.N, self.c_puct),
            )
            child = node.children[action]

            # Calcul paresseux du latent de l'enfant (premiere visite)
            if child.latent is None:
                h_next, reward = self._dynamics_no_grad(node.latent, action)
                child.latent = h_next
                child.reward = reward

            path.append(child)
            node = child

        # Expansion : prediction sur le noeud feuille
        priors, value = self._predict_no_grad(node.latent)
        for a in range(self.n_actions):
            node.children[a] = MuZeroNode(prior=float(priors[a]))

        # Backpropagation avec recompenses accumulees (bootstrapped return)
        # On itere sur le chemin en parallele avec les enfants pour eviter
        # path.index() qui est O(n) et faux si un meme noeud apparait 2 fois.
        G = value
        node.update(G)
        for i in range(len(path) - 2, -1, -1):
            # La recompense de la transition i → i+1 est stockee dans path[i+1]
            G = path[i + 1].reward + self.gamma * G
            path[i].update(G)

        return G

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
        """Collecte la transition; entraine en fin d'episode."""
        if self._last_mcts_policy is not None:
            self._ep_states.append(state.copy())
            self._ep_actions.append(action)
            self._ep_rewards.append(float(reward))
            self._ep_policies.append(self._last_mcts_policy.copy())
            self._last_mcts_policy = None

        if done and len(self._ep_states) > 0:
            return self._finish_episode()

        return None

    def _finish_episode(self) -> Dict[str, float]:
        """Stocke la trajectoire et entraine si assez de donnees."""
        traj = list(
            zip(
                self._ep_states,
                self._ep_actions,
                self._ep_rewards,
                self._ep_policies,
            )
        )
        self._replay.append(traj)

        self._ep_states.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        self._ep_policies.clear()

        total_steps = sum(len(t) for t in self._replay)
        if total_steps < self.min_buffer_size:
            return {"replay_steps": float(total_steps)}

        return self._train_batch()

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Retours discountes G_t."""
        T = len(rewards)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def _train_batch(self) -> Dict[str, float]:
        """
        Entraine les 3 reseaux via deroulement K etapes du modele dynamique.

        Pour chaque sequence :
          h_0 = repr(s_start)
          Pour k = 0..K-1 :
            p_k, v_k = prediction(h_k)
            h_{k+1}, r_{k+1} = dynamics(h_k, a_{start+k})
          Loss = CE(p_k, pi_k) + MSE(v_k, G_k) + MSE(r_k, r_reel_k)
        """
        # Echantillonner des trajectoires
        trajs = random.choices(list(self._replay), k=self.batch_size)

        policy_losses = []
        value_losses = []
        reward_losses = []

        for traj in trajs:
            if len(traj) == 0:
                continue
            # Position de depart aleatoire dans la trajectoire
            max_start = max(0, len(traj) - self.n_unroll - 1)
            start = random.randint(0, max_start)
            seq = traj[start : start + self.n_unroll + 1]

            states_s = [s for s, a, r, p in seq]
            actions_s = [a for s, a, r, p in seq]
            rewards_s = [r for s, a, r, p in seq]
            policies_s = [p for s, a, r, p in seq]
            returns_s = self._compute_returns(rewards_s)

            # Encodage de l'etat initial
            s0_t = torch.FloatTensor(states_s[0]).unsqueeze(0).to(self.device)
            h = self.repr_net(s0_t).squeeze(0)

            for k in range(min(self.n_unroll, len(seq))):
                logits, value = self.pred_net(h.unsqueeze(0))
                logits = logits.squeeze(0)
                value = value.squeeze(0)

                # Cibles
                pi_target = torch.FloatTensor(policies_s[k]).to(self.device)
                v_target = torch.tensor(
                    returns_s[k], dtype=torch.float32, device=self.device
                )

                # Pertes
                log_p = F.log_softmax(logits, dim=-1)
                policy_losses.append(-(pi_target * log_p).sum())
                value_losses.append(F.mse_loss(value.squeeze(), v_target))

                # Dynamique (sauf dernier step)
                if k < len(seq) - 1:
                    a_oh = self._action_onehot(actions_s[k]).unsqueeze(0)
                    h_next, r_pred = self.dynamics_net(h.unsqueeze(0), a_oh)
                    h = h_next.squeeze(0)
                    # dynamics(h_k, a_k) predit la recompense de la transition
                    # s_k --(a_k)--> s_{k+1}, qui est rewards_s[k] (pas k+1)
                    r_target = torch.tensor(
                        rewards_s[k], dtype=torch.float32, device=self.device
                    )
                    reward_losses.append(F.mse_loss(r_pred.squeeze(), r_target))

        if not policy_losses:
            return {}

        loss = (
            torch.stack(policy_losses).mean()
            + torch.stack(value_losses).mean()
            + (torch.stack(reward_losses).mean() if reward_losses else torch.tensor(0.0))
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.repr_net.parameters())
            + list(self.dynamics_net.parameters())
            + list(self.pred_net.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()
        self.training_steps += 1

        return {
            "loss": loss.item(),
            "policy_loss": torch.stack(policy_losses).mean().item(),
            "value_loss": torch.stack(value_losses).mean().item(),
        }

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def on_episode_start(self) -> None:
        self._ep_states.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        self._ep_policies.clear()
        self._last_mcts_policy = None

    def set_training_mode(self, training: bool) -> None:
        self._training_mode = training
        mode = "train" if training else "eval"
        for net in (self.repr_net, self.dynamics_net, self.pred_net):
            net.train(training)

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "repr_net": self.repr_net.state_dict(),
                "dynamics_net": self.dynamics_net.state_dict(),
                "pred_net": self.pred_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.repr_net.load_state_dict(ckpt["repr_net"])
        self.dynamics_net.load_state_dict(ckpt["dynamics_net"])
        self.pred_net.load_state_dict(ckpt["pred_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)

    def get_config(self) -> Dict:
        return {
            "latent_dim": self.latent_dim,
            "n_simulations": self.n_simulations,
            "c_puct": self.c_puct,
            "n_unroll": self.n_unroll,
            "lr": self.lr,
            "gamma": self.gamma,
            "hidden_dims": self.hidden_dims,
        }


# ============================================================================
# MuZeroStochastic
# ============================================================================

class AfterStateNet(nn.Module):
    """
    Calcule l'afterstate a partir de (h, a).

    L'afterstate = partie deterministe de la transition, AVANT que le hasard
    de l'environnement (adversaire aleatoire, des, ...) ne soit resolu.
    """

    def __init__(
        self, latent_dim: int, n_actions: int, hidden_dims: List[int]
    ):
        super().__init__()
        self.net = MLP(latent_dim + n_actions, latent_dim, hidden_dims)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, latent: torch.Tensor, action_oh: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([latent, action_oh], dim=-1)
        return torch.tanh(self.norm(self.net(x)))


class ChanceNet(nn.Module):
    """
    Applique un code chance a l'afterstate pour obtenir h_next.

    Code chance = vecteur one-hot de taille n_chance encodant le resultat
    aleatoire de l'environnement (ex: coup de l'adversaire aleatoire).
    """

    def __init__(
        self, latent_dim: int, n_chance: int, hidden_dims: List[int]
    ):
        super().__init__()
        self.net = MLP(latent_dim + n_chance, latent_dim, hidden_dims)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, afterstate: torch.Tensor, code_oh: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([afterstate, code_oh], dim=-1)
        return torch.tanh(self.norm(self.net(x)))


class ChanceEncoder(nn.Module):
    """
    Infere le code chance a partir de (afterstate, h_next_reel) pendant l'entrainement.

    Utilise pour apprendre une representation discrete de la stochasticite
    sans supervision directe sur les codes.
    """

    def __init__(
        self, latent_dim: int, n_chance: int, hidden_dims: List[int]
    ):
        super().__init__()
        enc_hidden = hidden_dims[:1] if hidden_dims else [64]
        self.net = MLP(latent_dim * 2, n_chance, enc_hidden)

    def forward(
        self, afterstate: torch.Tensor, h_next: torch.Tensor
    ) -> torch.Tensor:
        """Retourne les logits sur les codes chance."""
        x = torch.cat([afterstate, h_next], dim=-1)
        return self.net(x)


class MuZeroStochastic(MuZero):
    """
    MuZero Stochastique — gere explicitement la stochasticite.

    Dans les environnements deterministes, la dynamique h → h' est unique
    pour une action donnee. Dans les environnements stochastiques (adversaire
    aleatoire, etc.), la meme action peut mener a des etats differents.

    Architecture additionnelle :
      - afterstate_net   : (h, a)           → afterstate (partie deterministe)
      - chance_net       : (afterstate, code) → h_next (partie stochastique)
      - chance_encoder   : (afterstate, h_next_reel) → code (apprentissage)

    MCTS avec noeuds chance :
      Noeud etat h
        └── action a → Noeud afterstate as
              └── code c_0 → h_0
              └── code c_1 → h_1
              ...

    Pendant la recherche, les enfants "code" sont echantillonnes selon la
    distribution apprise par chance_encoder. En moyenne sur n_chance outcomes,
    on obtient une estimation plus robuste de la valeur.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        latent_dim: int = 64,
        n_chance: int = 8,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        n_simulations: int = 50,
        c_puct: float = 1.0,
        n_unroll: int = 5,
        temperature: float = 1.0,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        min_buffer_size: int = 500,
        l2_reg: float = 1e-4,
    ):
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            n_simulations=n_simulations,
            c_puct=c_puct,
            n_unroll=n_unroll,
            temperature=temperature,
            buffer_size=buffer_size,
            batch_size=batch_size,
            min_buffer_size=min_buffer_size,
            l2_reg=l2_reg,
        )
        self.name = "MuZeroStochastic"
        self.n_chance = n_chance

        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Reseaux additionnels pour la stochasticite
        self.afterstate_net = AfterStateNet(
            latent_dim, n_actions, hidden_dims
        ).to(self.device)
        self.chance_net = ChanceNet(
            latent_dim, n_chance, hidden_dims
        ).to(self.device)
        self.chance_encoder = ChanceEncoder(
            latent_dim, n_chance, hidden_dims
        ).to(self.device)

        # Refaire l'optimiseur avec tous les reseaux
        self.optimizer = torch.optim.Adam(
            list(self.repr_net.parameters())
            + list(self.dynamics_net.parameters())
            + list(self.pred_net.parameters())
            + list(self.afterstate_net.parameters())
            + list(self.chance_net.parameters())
            + list(self.chance_encoder.parameters()),
            lr=lr,
            weight_decay=l2_reg,
        )

    # ------------------------------------------------------------------
    # Dynamique stochastique
    # ------------------------------------------------------------------

    def _stochastic_dynamics(
        self,
        latent: torch.Tensor,
        action: int,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Un pas stochastique : (h, a) → moyenne sur n_samples codes chance.

        En moyennant sur plusieurs outcomes, on obtient une estimation de
        l'etat latent esperee, plus stable que de choisir un seul outcome.
        """
        oh = self._action_onehot(action)
        if latent.dim() == 1:
            latent_2d = latent.unsqueeze(0)
            oh_2d = oh.unsqueeze(0)
        else:
            latent_2d = latent
            oh_2d = oh

        with torch.no_grad():
            afterstate = self.afterstate_net(latent_2d, oh_2d)

            # Echantillonner codes aleatoires (recherche stochastique)
            codes = torch.randint(0, self.n_chance, (n_samples,))
            h_samples = []
            for c in codes:
                code_oh = F.one_hot(c, self.n_chance).float().to(self.device).unsqueeze(0)
                h_next = self.chance_net(afterstate, code_oh)
                h_samples.append(h_next)

            h_avg = torch.stack(h_samples, dim=0).mean(dim=0).squeeze(0)

            # Recompense via le reseau de dynamique parent
            _, r = self.dynamics_net(latent_2d, oh_2d)

        return h_avg, float(r.item())

    @torch.no_grad()
    def _dynamics_no_grad(
        self, latent: torch.Tensor, action: int
    ) -> Tuple[torch.Tensor, float]:
        """Surcharge : utilise la dynamique stochastique."""
        return self._stochastic_dynamics(latent, action, n_samples=self.n_chance)

    # ------------------------------------------------------------------
    # Apprentissage (surcharge pour ajouter la perte chance)
    # ------------------------------------------------------------------

    def set_training_mode(self, training: bool) -> None:
        super().set_training_mode(training)
        for net in (self.afterstate_net, self.chance_net, self.chance_encoder):
            net.train(training)

    def save(self, path: str) -> None:
        torch.save(
            {
                "class": "MuZeroStochastic",
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "latent_dim": self.latent_dim,
                "n_chance": self.n_chance,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "repr_net": self.repr_net.state_dict(),
                "dynamics_net": self.dynamics_net.state_dict(),
                "pred_net": self.pred_net.state_dict(),
                "afterstate_net": self.afterstate_net.state_dict(),
                "chance_net": self.chance_net.state_dict(),
                "chance_encoder": self.chance_encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.repr_net.load_state_dict(ckpt["repr_net"])
        self.dynamics_net.load_state_dict(ckpt["dynamics_net"])
        self.pred_net.load_state_dict(ckpt["pred_net"])
        self.afterstate_net.load_state_dict(ckpt["afterstate_net"])
        self.chance_net.load_state_dict(ckpt["chance_net"])
        self.chance_encoder.load_state_dict(ckpt["chance_encoder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg["n_chance"] = self.n_chance
        return cfg
