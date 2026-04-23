"""
REINFORCE et ses variantes (Policy Gradient).

Progression pedagogique — chaque classe reduit la variance de la precedente :

1. REINFORCE
   Gradient Monte-Carlo pur. Collecte une trajectoire complete, calcule
   les retours G_t, et met a jour par gradient ascendant sur log pi * G_t.
   Variance elevee car G_t depend de tous les futurs aleatoires.

2. REINFORCEWithMeanBaseline
   Soustrait la moyenne des retours avant la mise a jour.
   G_t - mean(G) ne change pas l'esperance du gradient (baseline sans biais)
   mais reduit significativement la variance.

3. REINFORCEWithCriticBaseline
   Un second reseau V(s; phi) apprend la valeur d'etat.
   L'avantage A_t = G_t - V(s_t) remplace G_t : le critique "retire" la
   part previsible du retour, ne laissant que la part informationnelle.

4. PPO (Proximal Policy Optimization, A2C style)
   Ajoute un clip sur le ratio pi_new/pi_old pour limiter la taille
   des mises a jour et eviter les effondrement de politique.
   Mise a jour sur n_epochs passes sur la meme trajectoire.

Differences cles avec DQN :
- Pas de reseau Q : le reseau EST la politique (sortie = distribution)
- Mise a jour en fin d'episode (trajectoire complete), pas step-by-step
- Gradient stochastique via le log-trick : grad log pi(a|s) * G_t

References :
- Sutton & Barto (2018), Ch. 13 (Policy Gradient Methods)
- Williams (1992) "Simple Statistical Gradient-Following Algorithms"
- Mnih et al. (2016) "Asynchronous Methods for Deep RL" (A3C/A2C baseline)
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Optional, Dict

from deeprl.agents.base import Agent
from deeprl.networks.mlp import MLP


# ============================================================================
# 1. REINFORCE
# ============================================================================

class REINFORCE(Agent):
    """
    REINFORCE — Gradient de politique Monte-Carlo.

    L'idee fondamentale : si une action a conduit a un bon retour,
    augmenter sa probabilite. Si elle a conduit a un mauvais retour,
    la diminuer.

    Gradient de la politique (Williams, 1992) :
        grad J(theta) = E[ grad log pi(a_t|s_t; theta) * G_t ]

    ou G_t = sum_{t'>=t} gamma^{t'-t} * r_{t'} est le retour disconte.

    Mise a jour (gradient ascendant) :
        theta <- theta + alpha * grad J(theta)
        soit : loss = - mean( log pi(a_t|s_t) * G_t )   [minimise]

    La mise a jour a lieu UNE FOIS par episode, apres avoir collecte
    la trajectoire complete (Monte-Carlo).

    Avantage sur DQN :
    - Peut apprendre des politiques stochastiques
    - Pas de biais de maximisation
    - Supporte naturellement les espaces d'actions continus

    Inconvenient :
    - Variance tres elevee de l'estimateur de gradient
    - Necessite des episodes complets (pas de TD-learning)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Optional[List[int]] = None,
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
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="REINFORCE"
        )

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.gamma = gamma
        self.lr = lr
        self.hidden_dims = hidden_dims

        # Reseau de politique : state -> logits (pas de softmax en sortie)
        # Le softmax est applique manuellement dans act() pour masquer les
        # actions invalides AVANT la normalisation.
        self.policy_net = MLP(state_dim, n_actions, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self._training = True

        # Buffers de trajectoire (remplis pendant l'episode, vides apres MAJ)
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

        # Pour stocker le log_prob de l'action courante entre act() et learn()
        self._last_log_prob: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(self, state, available_actions=None, training=True, **kwargs):
        """
        Choisit une action selon la distribution de politique courante.

        - Mode entrainement : echantillonne depuis pi(.|s) (stochastique)
        - Mode evaluation : action la plus probable (deterministe)

        Le masquage des actions invalides est applique sur les logits
        AVANT le softmax pour que la distribution reste valide.
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad() if not (training and self._training) else torch.enable_grad():
            logits = self.policy_net(state_t).squeeze(0)  # (n_actions,)

        # Masquer les actions invalides : mettre -inf sur les logits invalides
        # pour qu'elles aient proba 0 apres softmax
        mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
        for a in available_actions:
            mask[a] = 0.0
        masked_logits = logits + mask

        if training and self._training:
            # Echantillonnage stochastique depuis la politique
            dist = Categorical(logits=masked_logits)
            action_t = dist.sample()
            self._last_log_prob = dist.log_prob(action_t)
            return int(action_t.item())
        else:
            # Exploitation pure : argmax des probabilites
            self._last_log_prob = None
            return int(masked_logits.argmax().item())

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        Collecte la transition. Met a jour la politique en fin d'episode.

        A chaque step :
        - Stocke (log_prob, reward) dans les buffers de trajectoire

        Quand done=True :
        - Calcule les retours G_t = sum gamma^k * r_{t+k}
        - Met a jour les poids : loss = -mean(log_probs * G_t)
        - Vide les buffers
        """
        if self._last_log_prob is not None:
            self._log_probs.append(self._last_log_prob)
            self._rewards.append(float(reward))
            self._last_log_prob = None

        if done and len(self._log_probs) > 0:
            info = self._update_policy()
            self._log_probs = []
            self._rewards = []
            return info

        return None

    def _compute_returns(self) -> torch.Tensor:
        """
        Calcule les retours discountes G_t pour chaque step de la trajectoire.

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
            = r_t + gamma * G_{t+1}

        On remonte depuis la fin (calcul recursif efficace).

        Returns:
            Tensor de forme (T,) avec les retours normalises
        """
        T = len(self._rewards)
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def _update_policy(self) -> Dict[str, float]:
        """
        Mise a jour des poids par gradient ascendant.

        loss = -mean( log pi(a_t|s_t) * G_t )

        Le signe negatif transforme l'ascendant en descente de gradient
        (convention PyTorch : minimise la loss).
        """
        returns = self._compute_returns()

        log_probs_t = torch.stack(self._log_probs)  # (T,)

        # REINFORCE loss : -E[log pi(a|s) * G]
        loss = -(log_probs_t * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        return {
            "loss": loss.item(),
            "mean_return": returns.mean().item(),
        }

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def on_episode_start(self):
        """Vide les buffers au debut d'un episode (securite max_steps)."""
        self._log_probs = []
        self._rewards = []
        self._last_log_prob = None

    def set_training_mode(self, training: bool):
        self._training = training
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
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
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)


# ============================================================================
# 2. REINFORCE WITH MEAN BASELINE
# ============================================================================

class REINFORCEWithMeanBaseline(REINFORCE):
    """
    REINFORCE avec baseline = moyenne des retours de l'episode.

    Probleme de REINFORCE pur : les retours G_t sont toujours positifs
    dans certains environnements, donc le gradient pousse toujours dans
    le meme sens, avec une variance elevee.

    Solution (Sutton & Barto, Ch. 13.4) : soustraire une baseline b(s)
    qui ne depend pas de l'action. Cela ne change pas l'esperance du
    gradient (la baseline est "sans biais") mais reduit la variance :

        grad J = E[ grad log pi(a|s) * (G_t - b) ]

    La baseline la plus simple : b = mean(G_t) sur l'episode courant.

    Avantage : aucun parametre supplementaire, reduction de variance
    immediatement efficace sur les episodes courts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "REINFORCEWithMeanBaseline"

    def _update_policy(self) -> Dict[str, float]:
        """
        Identique a REINFORCE mais avec G_t - mean(G) comme signal.

        La soustraction de la moyenne centre les retours autour de 0 :
        - G_t > mean → renforce l'action (signal positif)
        - G_t < mean → inhibe l'action (signal negatif)
        """
        returns = self._compute_returns()

        # Baseline = moyenne des retours de cet episode
        baseline = returns.mean()
        advantages = returns - baseline

        log_probs_t = torch.stack(self._log_probs)
        loss = -(log_probs_t * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        return {
            "loss": loss.item(),
            "mean_return": returns.mean().item(),
            "baseline": baseline.item(),
        }


# ============================================================================
# 3. REINFORCE WITH CRITIC BASELINE
# ============================================================================

class REINFORCEWithCriticBaseline(REINFORCE):
    """
    REINFORCE avec baseline apprise par un critique V(s; phi).

    Le critique apprend a predire la valeur d'etat V(s) = E[G_t | s_t = s].
    L'avantage A_t = G_t - V(s_t) mesure si l'action etait MEILLEURE
    ou MOINS BONNE que prevu.

    Deux reseaux, deux losses :
    - Politique theta : loss_pi = -mean(log pi(a|s) * A_t.detach())
    - Critique phi    : loss_v  = MSE(V(s_t), G_t)

    Le detach() sur A_t est crucial : le gradient de la policy ne doit
    pas traverser le critique (ce sont deux optimiseurs independants).

    Avantage sur mean baseline : le critique s'adapte etat par etat,
    donc la reduction de variance est plus fine et augmente avec le temps.

    Note : A_t.detach() → le critique est optimise separement
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
    ):
        """
        Args:
            lr_critic: Learning rate du critique (peut differer de la politique)
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions,
            hidden_dims=hidden_dims, lr=lr, gamma=gamma,
        )
        self.name = "REINFORCEWithCriticBaseline"
        self.lr_critic = lr_critic

        # Reseau critique : state -> V(s) scalaire
        self.value_net = MLP(state_dim, 1, self.hidden_dims).to(self.device)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr_critic
        )
        self.value_loss_fn = nn.MSELoss()

        # Buffer supplementaire pour les valeurs d'etat
        self._states: List[torch.Tensor] = []

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        Stocke aussi l'etat courant pour que le critique puisse s'entrainer.
        """
        if self._last_log_prob is not None:
            # Stocker l'etat pour le critique
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self._states.append(state_t)

        return super().learn(state, action, reward, next_state, done, **kwargs)

    def on_episode_start(self):
        """Vide tous les buffers."""
        super().on_episode_start()
        self._states = []

    def _update_policy(self) -> Dict[str, float]:
        """
        Double mise a jour : politique (avec avantage) et critique (MSE).

        1. Calculer G_t (retours discountes)
        2. Calculer V(s_t) pour tous les etats de la trajectoire
        3. Avantage : A_t = G_t - V(s_t)
        4. Politique : loss_pi = -mean(log pi * A_t.detach())
        5. Critique  : loss_v  = MSE(V(s_t), G_t)
        """
        returns = self._compute_returns()  # (T,)

        if len(self._states) != len(self._log_probs):
            # Securite : si buffers desynchronises, skip
            return {"loss": 0.0, "value_loss": 0.0}

        # Valeurs d'etat predites par le critique
        states_t = torch.cat(self._states, dim=0)   # (T, state_dim)
        values = self.value_net(states_t).squeeze(1) # (T,)

        # Avantage : A_t = G_t - V(s_t)
        # detach() : le gradient de la politique ne remonte pas dans le critique
        advantages = (returns - values.detach())

        # ---- Mise a jour de la politique ----
        log_probs_t = torch.stack(self._log_probs)  # (T,)
        policy_loss = -(log_probs_t * advantages).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- Mise a jour du critique ----
        value_loss = self.value_loss_fn(values, returns.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        self.training_steps += 1

        return {
            "loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

    def set_training_mode(self, training: bool):
        super().set_training_mode(training)
        if training:
            self.value_net.train()
        else:
            self.value_net.eval()

    def save(self, path: str):
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "lr_critic": self.lr_critic,
                "gamma": self.gamma,
                "policy_net": self.policy_net.state_dict(),
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)


# ============================================================================
# 4. PPO (Proximal Policy Optimization, A2C style)
# ============================================================================

class PPO(REINFORCEWithCriticBaseline):
    """
    Proximal Policy Optimization — style A2C avec clipping.

    Probleme de REINFORCE+Critic : une grande mise a jour peut degader
    drastiquement la politique (falaise dans l'espace des parametres).
    REINFORCE ne peut pas se "rattraper" car la trajectoire est jetee.

    Solution PPO (Schulman et al., 2017) : clipper le ratio pi_new/pi_old
    pour que la politique ne s'eloigne pas trop en un seul update.

    Objectif clippe :
        L_clip = E[ min( r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t ) ]
        ou r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)

    Pourquoi min ? Le clip seul pourrait favoriser des mises a jour dans
    un seul sens. Le min assure que l'objectif est pessimiste : on prend
    le pire des deux cas (non-clipe vs clipe).

    Loss totale :
        L = -L_clip + value_coef * L_value - entropy_coef * H(pi)

    Le terme d'entropie H(pi) encourage l'exploration en penalisant
    les politiques trop deterministes.

    Differrence avec REINFORCE :
    - n_epochs passes sur la meme trajectoire (meilleure sample efficiency)
    - Clip evite les grandes mises a jour destabilisantes
    - Entropie maintient l'exploration
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        """
        Args:
            clip_eps: Taille du clip sur le ratio (defaut: 0.2)
            n_epochs: Nombre de passes de MAJ sur une trajectoire
            entropy_coef: Coefficient du bonus d'entropie
            value_coef: Coefficient de la loss du critique
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions,
            hidden_dims=hidden_dims, lr=lr, lr_critic=lr_critic, gamma=gamma,
        )
        self.name = "PPO"

        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def _update_policy(self) -> Dict[str, float]:
        """
        Mise a jour PPO avec clipping et plusieurs epochs.

        1. Calculer G_t et A_t avec les anciens log_probs (frozen)
        2. Pour n_epochs iterations :
           a. Recalculer log_probs avec les poids courants
           b. ratio = exp(new_log_probs - old_log_probs)
           c. L_clip = min(ratio * A, clip(ratio, 1±ε) * A)
           d. Actualiser les poids
        """
        returns = self._compute_returns()

        if len(self._states) != len(self._log_probs):
            return {"loss": 0.0, "value_loss": 0.0}

        states_t = torch.cat(self._states, dim=0)         # (T, state_dim)
        actions_t = self._collect_actions()                # (T,)
        old_log_probs = torch.stack(self._log_probs).detach()  # (T,) frozen

        # Avantages calcules une seule fois (avec valeurs courantes)
        with torch.no_grad():
            values_old = self.value_net(states_t).squeeze(1)
        advantages = (returns - values_old).detach()

        # Normalisation des avantages (stabilite numerique)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.n_epochs):
            # ---- Recalculer log_probs avec poids courants ----
            logits = self.policy_net(states_t)              # (T, n_actions)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)        # (T,)
            entropy = dist.entropy().mean()                 # scalaire

            # ---- Ratio pi_new / pi_old ----
            # exp(log pi_new - log pi_old) = pi_new / pi_old
            ratio = torch.exp(new_log_probs - old_log_probs)  # (T,)

            # ---- Objectif clippe ----
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # ---- Loss du critique ----
            values_new = self.value_net(states_t).squeeze(1)
            value_loss = self.value_loss_fn(values_new, returns.detach())

            # ---- Loss totale ----
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.training_steps += 1

        return {
            "loss": total_policy_loss / self.n_epochs,
            "value_loss": total_value_loss / self.n_epochs,
            "entropy": total_entropy / self.n_epochs,
            "mean_return": returns.mean().item(),
        }

    def _collect_actions(self) -> torch.Tensor:
        """
        Retourne le tenseur des actions stockees dans le buffer d'episode.

        PPO doit recalculer log pi_new(a_t|s_t) pendant les n_epochs passes,
        ce qui requiert de connaitre les actions exactes prises. Elles sont
        accumulees dans self._actions par learn().
        """
        return torch.tensor(self._actions, dtype=torch.long, device=self.device)

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        Stocke aussi l'action pour PPO (necessaire pour recalculer le ratio).
        """
        # Initialiser le buffer d'actions si pas encore fait
        if not hasattr(self, '_actions'):
            self._actions: List[int] = []

        if self._last_log_prob is not None:
            self._actions.append(action)

        result = super().learn(state, action, reward, next_state, done, **kwargs)

        if done:
            self._actions = []

        return result

    def on_episode_start(self):
        """Vide tous les buffers y compris les actions."""
        super().on_episode_start()
        self._actions = []

    def save(self, path: str):
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "lr_critic": self.lr_critic,
                "gamma": self.gamma,
                "clip_eps": self.clip_eps,
                "n_epochs": self.n_epochs,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "policy_net": self.policy_net.state_dict(),
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )
