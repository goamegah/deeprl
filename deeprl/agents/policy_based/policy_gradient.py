"""
Policy Gradient — Methodes basees sur la politique directe.

Progression pedagogique — chaque classe ajoute UN concept :

1. REINFORCE (Williams, 1992)
   Apprend directement pi(a|s; theta) par gradient ascent sur le retour
   cumule. Mise a jour Monte Carlo : attend la fin de l'episode.

2. REINFORCEWithBaseline
   Soustrait la moyenne des retours comme baseline pour reduire la
   variance du gradient sans introduire de biais.
       gradient = sum log pi(at|st) * (Gt - mean(G))

3. REINFORCEWithCritic
   Remplace la baseline constante par une fonction de valeur V(s; phi)
   apprise par un second reseau (le critique).
       advantage = Gt - V(st; phi)   [Actor-Critic Monte Carlo]

4. PPO (Proximal Policy Optimization, Schulman et al. 2017)
   Ajoute une contrainte de clipping sur le ratio pi_new/pi_old pour
   empecher des mises a jour trop grandes qui destabilisent la politique.
       L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

References :
- Williams (1992) "Simple statistical gradient-following algorithms"
- Sutton & Barto (2018), Ch. 13 — Policy Gradient Methods
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Mnih et al. (2016) "Asynchronous Methods for Deep RL" (A3C/A2C)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from deeprl.agents.base import Agent
from deeprl.networks.mlp import MLP


# ============================================================================
# 1. REINFORCE
# ============================================================================

class REINFORCE(Agent):
    """
    REINFORCE — Monte Carlo Policy Gradient (Williams, 1992).

    Apprend la politique pi(a|s; theta) directement, sans passer par
    une fonction de valeur intermediaire.

    A chaque etape t, l'agent enregistre (st, at, rt). En fin d'episode,
    il calcule le retour cumule Gt = sum_{k>=t} gamma^(k-t) * r_{k+1}
    et met a jour le reseau par gradient ascent :

        theta <- theta + alpha * sum_t grad_theta log pi(at|st) * Gt

    Probleme : Gt a une variance elevee (trajectoire complete). Les
    variantes suivantes reduisent cette variance.

    Differences avec DQN :
    - DQN apprend Q(s,a) et derive la politique par argmax
    - REINFORCE apprend pi(a|s) directement par sampling
    - DQN : mise a jour a chaque step / REINFORCE : fin d'episode
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        """
        Args:
            state_dim: Dimension du vecteur d'etat
            n_actions: Nombre d'actions possibles
            hidden_dims: Couches cachees du MLP (defaut: [64, 64])
            lr: Learning rate (Adam)
            gamma: Facteur d'actualisation
        """
        super().__init__(state_dim=state_dim, n_actions=n_actions, name="REINFORCE")

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.gamma = gamma
        self.lr = lr
        self.hidden_dims = hidden_dims

        # Reseau de politique : pi(a|s; theta) -> proba sur les actions
        self.policy_net = MLP(state_dim, n_actions, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self._training = True
        # Buffer d'episode : accumule les transitions jusqu'a done=True
        self._log_probs = []
        self._rewards = []
        self._last_log_prob = None

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(self, state, available_actions=None, training=True, **kwargs):
        """
        Echantillonne une action depuis la politique.

        En entrainement : sampling stochastique depuis pi(a|s).
        En evaluation  : action la plus probable (greedy).
        """
        is_training = training and self._training
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if not is_training:
            with torch.no_grad():
                logits = self.policy_net(state_t).squeeze(0)
                probs = self._mask_and_normalize(logits, available_actions)
                return int(probs.argmax().item())

        logits = self.policy_net(state_t).squeeze(0)
        probs = self._mask_and_normalize(logits, available_actions)
        dist = Categorical(probs)
        action = dist.sample()
        self._last_log_prob = dist.log_prob(action)
        return action.item()

    def _mask_and_normalize(self, logits, available_actions):
        """Applique softmax + masque des actions invalides."""
        probs = torch.softmax(logits, dim=-1)
        if available_actions is not None:
            mask = torch.zeros(self.n_actions, device=self.device)
            mask[list(available_actions)] = 1.0
            probs = probs * mask
            probs = probs / (probs.sum() + 1e-8)
        return probs

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        Accumule les transitions. Met a jour uniquement en fin d'episode.

        Retourne les metriques seulement quand done=True.
        """
        if self._last_log_prob is not None:
            self._log_probs.append(self._last_log_prob)
            self._rewards.append(reward)
            self._last_log_prob = None

        if not done:
            return None

        return self._update()

    def _compute_returns(self):
        """Calcule les retours cumules Gt = sum_{k>=t} gamma^(k-t) * r_k."""
        G = 0.0
        returns = []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def _update(self):
        """Gradient ascent sur le retour cumule."""
        if not self._log_probs:
            self._clear_buffer()
            return None

        returns = self._compute_returns()
        log_probs = torch.stack(self._log_probs)

        # Perte REINFORCE : -E[log pi(a|s) * Gt]
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        mean_return = returns[0].item()
        self._clear_buffer()
        self.training_steps += 1

        return {"loss": loss.item(), "return": mean_return}

    def _clear_buffer(self):
        self._log_probs.clear()
        self._rewards.clear()
        self._last_log_prob = None

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def on_episode_start(self):
        self._clear_buffer()

    def set_training_mode(self, training: bool):
        self._training = training
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "class": self.__class__.__name__,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "gamma": self.gamma,
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)


# ============================================================================
# 2. REINFORCE WITH MEAN BASELINE
# ============================================================================

class REINFORCEWithBaseline(REINFORCE):
    """
    REINFORCE avec baseline constante — soustrait la moyenne des retours.

    Probleme de REINFORCE de base : la variance de Gt est elevee car
    elle depend de toute la trajectoire future. Cela ralentit l'apprentissage.

    Solution : soustraire une baseline b qui ne depend pas de l'action :
        gradient = sum_t grad_theta log pi(at|st) * (Gt - b)

    Baseline choisie : b = mean(G) sur l'episode courant.

    Propriete : la baseline ne cree pas de biais car E[grad log pi * b] = 0
    (le gradient d'une constante par rapport a la politique est nul).

    Effet : les retours au-dessus de la moyenne sont renforces,
    ceux en dessous sont penalises.
    """

    def __init__(self, state_dim, n_actions, hidden_dims=None, lr=1e-3, gamma=0.99):
        super().__init__(state_dim, n_actions, hidden_dims, lr, gamma)
        self.name = "REINFORCE_Baseline"

    def _update(self):
        """Gradient ascent avec baseline = mean(G)."""
        if not self._log_probs:
            self._clear_buffer()
            return None

        returns = self._compute_returns()
        log_probs = torch.stack(self._log_probs)

        # Baseline : moyenne des retours de l'episode
        baseline = returns.mean()
        advantages = returns - baseline

        loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        mean_return = returns[0].item()
        self._clear_buffer()
        self.training_steps += 1

        return {
            "loss": loss.item(),
            "return": mean_return,
            "baseline": baseline.item(),
        }


# ============================================================================
# 3. REINFORCE WITH CRITIC (Actor-Critic Monte Carlo)
# ============================================================================

class REINFORCEWithCritic(REINFORCE):
    """
    REINFORCE avec baseline apprise par un critique — Actor-Critic.

    Limite des baselines constantes : elles ne dependent pas de l'etat.
    Une meilleure baseline est V(st), la valeur esperee de l'etat courant.

    Architecture :
    - Acteur  : pi(a|s; theta)   [meme reseau que REINFORCE]
    - Critique: V(s; phi)        [nouveau reseau, predit le retour attendu]

    Mise a jour :
        advantage_t = Gt - V(st; phi)          [erreur de prediction]
        Actor  loss = -sum log pi(at|st) * advantage_t
        Critic loss = MSE(V(st; phi), Gt)      [regression vers Gt]

    Le critique reduit la variance de l'acteur en fournissant une
    reference adaptee a chaque etat (pas juste une moyenne globale).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        critic_coef: float = 0.5,
    ):
        """
        Args:
            lr_actor:    Learning rate de l'acteur
            lr_critic:   Learning rate du critique
            critic_coef: Poids de la perte du critique dans la loss totale
        """
        super().__init__(state_dim, n_actions, hidden_dims, lr_actor, gamma)
        self.name = "REINFORCE_Critic"
        self.critic_coef = critic_coef
        self.lr_critic = lr_critic

        # Reseau critique : V(s; phi) -> scalaire
        self.critic_net = MLP(state_dim, 1, hidden_dims).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=lr_critic
        )
        self.critic_loss_fn = nn.MSELoss()

        # Stocke aussi les etats pour evaluer V(st)
        self._states = []

    def learn(self, state, action, reward, next_state, done, **kwargs):
        if self._last_log_prob is not None:
            self._log_probs.append(self._last_log_prob)
            self._rewards.append(reward)
            self._states.append(state)
            self._last_log_prob = None

        if not done:
            return None

        return self._update()

    def _update(self):
        """Mise a jour actor + critic avec advantage = Gt - V(st)."""
        if not self._log_probs:
            self._clear_buffer()
            return None

        returns = self._compute_returns()
        log_probs = torch.stack(self._log_probs)
        states_t = torch.FloatTensor(np.array(self._states)).to(self.device)

        # Valeurs predites par le critique : V(st; phi)
        values = self.critic_net(states_t).squeeze(1)

        # Avantages : Gt - V(st) — detach pour ne pas backprop dans le critique via l'acteur
        advantages = returns - values.detach()

        # Perte acteur
        actor_loss = -(log_probs * advantages).mean()

        # Perte critique : MSE(V(st), Gt)
        critic_loss = self.critic_loss_fn(values, returns)

        total_loss = actor_loss + self.critic_coef * critic_loss

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.critic_optimizer.step()

        mean_return = returns[0].item()
        self._clear_buffer()
        self.training_steps += 1

        return {
            "loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "return": mean_return,
            "value": values.mean().item(),
        }

    def _clear_buffer(self):
        super()._clear_buffer()
        self._states.clear()

    def set_training_mode(self, training: bool):
        super().set_training_mode(training)
        if training:
            self.critic_net.train()
        else:
            self.critic_net.eval()

    def save(self, path):
        torch.save({
            "class": self.__class__.__name__,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "lr_critic": self.lr_critic,
            "gamma": self.gamma,
            "critic_coef": self.critic_coef,
            "policy_net": self.policy_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.critic_net.load_state_dict(ckpt["critic_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)


# ============================================================================
# 4. PPO (Proximal Policy Optimization — A2C style)
# ============================================================================

class PPO(REINFORCEWithCritic):
    """
    PPO — Proximal Policy Optimization (Schulman et al., 2017).

    Probleme d'Actor-Critic : une mise a jour trop grande peut faire
    sortir la politique de sa zone de stabilite et degrader les perfs
    de facon irreversible (effondrement de politique).

    Solution PPO : contraindre la mise a jour en clippant le ratio
    entre la nouvelle et l'ancienne politique :

        r_t(theta) = pi_new(at|st) / pi_old(at|st)

        L_CLIP = E[min(r_t * At, clip(r_t, 1-eps, 1+eps) * At)]

    Quand r_t > 1+eps (mise a jour trop agressive dans le bon sens),
    le gradient est bloque => stabilite garantie.

    Avantages vs REINFORCE/A2C :
    - Mise a jour plus stable (pas d'effondrement de politique)
    - Peut reutiliser les transitions plusieurs fois (ppo_epochs)
    - Meilleur compromis exploration/exploitation via bonus d'entropie

    Implementation : A2C style (un seul worker, mise a jour par episode).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        Args:
            clip_eps:     Seuil de clipping (epsilon dans le papier, defaut 0.2)
            ppo_epochs:   Nombre de passes sur le batch par episode
            entropy_coef: Bonus d'entropie pour encourager l'exploration
        """
        super().__init__(
            state_dim, n_actions, hidden_dims,
            lr_actor, lr_critic, gamma, critic_coef,
        )
        self.name = "PPO"
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef

        # Stocke les actions et masques pour recalculer log_probs dans les epochs
        self._actions = []
        self._available_actions_list = []

    def act(self, state, available_actions=None, training=True, **kwargs):
        self._last_available = available_actions
        return super().act(state, available_actions, training, **kwargs)

    def learn(self, state, action, reward, next_state, done, **kwargs):
        if self._last_log_prob is not None:
            self._log_probs.append(self._last_log_prob.detach())  # log_prob ancienne politique
            self._rewards.append(reward)
            self._states.append(state)
            self._actions.append(action)
            self._available_actions_list.append(self._last_available)
            self._last_log_prob = None

        if not done:
            return None

        return self._update()

    def _update(self):
        """PPO : multiple epochs sur le batch avec clipped objective."""
        if not self._log_probs:
            self._clear_buffer()
            return None

        returns = self._compute_returns()
        # log_probs de l'ancienne politique (fixes pendant les epochs)
        old_log_probs = torch.stack(self._log_probs)
        states_t = torch.FloatTensor(np.array(self._states)).to(self.device)
        actions_t = torch.LongTensor(self._actions).to(self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.ppo_epochs):
            # Recalculer log_probs et valeurs avec la politique COURANTE
            logits = self.policy_net(states_t)
            values = self.critic_net(states_t).squeeze(1)

            # Recalculer les log_probs pour les actions choisies
            new_log_probs = []
            entropies = []
            for i, avail in enumerate(self._available_actions_list):
                probs_i = self._mask_and_normalize(logits[i], avail)
                dist_i = Categorical(probs_i)
                new_log_probs.append(dist_i.log_prob(actions_t[i]))
                entropies.append(dist_i.entropy())

            new_log_probs_t = torch.stack(new_log_probs)
            entropy_t = torch.stack(entropies).mean()

            # Advantages normalises pour la stabilite
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Ratio pi_new / pi_old en espace log pour la stabilite numerique
            ratio = torch.exp(new_log_probs_t - old_log_probs)

            # Objectif clippe PPO
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Perte critique
            critic_loss = self.critic_loss_fn(values, returns)

            # Loss totale avec bonus d'entropie (encourage l'exploration)
            loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_t

            self.optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=10.0)
            self.optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        mean_return = returns[0].item()
        self._clear_buffer()
        self.training_steps += 1

        return {
            "loss": (total_actor_loss + total_critic_loss) / self.ppo_epochs,
            "actor_loss": total_actor_loss / self.ppo_epochs,
            "critic_loss": total_critic_loss / self.ppo_epochs,
            "return": mean_return,
        }

    def _clear_buffer(self):
        super()._clear_buffer()
        self._actions.clear()
        self._available_actions_list.clear()

    def save(self, path):
        torch.save({
            "class": self.__class__.__name__,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "lr_critic": self.lr_critic,
            "gamma": self.gamma,
            "clip_eps": self.clip_eps,
            "ppo_epochs": self.ppo_epochs,
            "critic_coef": self.critic_coef,
            "entropy_coef": self.entropy_coef,
            "policy_net": self.policy_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
        }, path)
