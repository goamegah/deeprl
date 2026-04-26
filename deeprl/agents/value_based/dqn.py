"""
Deep Q-Learning et ses variantes.

Progression pedagogique — chaque classe ajoute UN concept :

1. DeepQLearning
   Remplace la Q-table par un reseau de neurones Q(s;theta) -> R^|A|.
   Mises a jour en ligne (semi-gradient). Instable car la cible
   utilise le meme reseau que la prediction.

2. DoubleDeepQLearning
   Ajoute un reseau cible theta^- copie periodiquement depuis theta.
   Decouple selection et evaluation pour reduire la surestimation.

3. DDQNWithExperienceReplay
   Ajoute un buffer de rejeu d'experience. Echantillonne des mini-batches
   uniformes pour casser les correlations temporelles.

4. DDQNWithPrioritizedExperienceReplay
   Les transitions a fort TD-error sont echantillonnees plus souvent.
   Corrige le biais avec Importance Sampling.

References :
- Sutton & Barto (2018), Ch. 9-10 (approximation de fonctions)
- Mnih et al. (2015) "Human-level control through deep RL"
- van Hasselt et al. (2016) "Deep RL with Double Q-learning"
- Schaul et al. (2016) "Prioritized Experience Replay"
"""

import numpy as np
import torch
import torch.nn as nn

from deeprl.agents.base import Agent
from deeprl.networks.mlp import MLP
from deeprl.memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


# ============================================================================
# 1. DEEP Q-LEARNING (DQN)
# ============================================================================

class DeepQLearning(Agent):
    """
    Deep Q-Learning — Approximation de Q par reseau de neurones.

    Remplace la table Q(s,a) de dimension finie par un reseau
    Q(s; theta) : R^d -> R^|A| qui generalise a des etats non visites.

    Politique : epsilon-greedy avec masquage des actions invalides.

    Mise a jour semi-gradient (Sutton & Barto, eq. 10.2) :
        theta <- theta - alpha * grad_theta [ Q(s,a;theta) - y ]^2
        ou y = r + gamma * max_a' Q(s', a'; theta)   [cible bootstrap]

    Limites :
    - La cible y depend de theta => cible mouvante => instabilite
    - Mises a jour en ligne => correlations entre transitions successives
    - max_a' Q(s',a';theta) surestime les Q-valeurs (biais de maximisation)

    => Voir DoubleDeepQLearning pour la correction.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Args:
            state_dim: Dimension du vecteur d'etat
            n_actions: Nombre d'actions possibles
            hidden_dims: Couches cachees du MLP (defaut: [64, 64])
            lr: Learning rate (Adam)
            gamma: Facteur d'actualisation
            epsilon_start: Epsilon initial (exploration)
            epsilon_end: Epsilon minimal
            epsilon_decay: Facteur de decroissance par episode
        """
        super().__init__(
            state_dim=state_dim, n_actions=n_actions, name="DeepQLearning"
        )

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Hyperparametres
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.hidden_dims = hidden_dims

        # Reseau Q
        self.q_net = MLP(state_dim, n_actions, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self._training = True
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def act(self, state, available_actions=None, training=True, **kwargs):
        """
        Choisit une action par epsilon-greedy.

        - Avec proba epsilon : action aleatoire (exploration)
        - Sinon : argmax_a Q(s, a) parmi les actions disponibles
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        use_training = training and self._training

        # Exploration epsilon-greedy
        if use_training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))

        # Exploitation : argmax Q(s, a)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        # Masquer les actions invalides
        masked_q = np.full(self.n_actions, -np.inf)
        for a in available_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        Mise a jour en ligne (une transition).

        Cible : y = r + gamma * max_a' Q(s', a'; theta)  [meme reseau]
        Perte : MSE( Q(s, a; theta), y )
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Q(s, a; theta)
        q_value = self.q_net(state_t)[0, action]

        # Cible y
        with torch.no_grad():
            if done:
                target = torch.tensor(
                    reward, device=self.device, dtype=torch.float32
                )
            else:
                next_q = self.q_net(next_state_t)

                # Masquer les actions invalides pour le next_state
                available_next = kwargs.get("available_actions_next")
                if available_next:
                    mask = torch.full(
                        (self.n_actions,), float("-inf"), device=self.device
                    )
                    for a in available_next:
                        mask[a] = 0.0
                    next_q = next_q + mask

                target = reward + self.gamma * next_q.max(dim=1)[0].squeeze()

        # Assurer la meme forme pour la loss
        loss = self.loss_fn(q_value.unsqueeze(0), target.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        # Clipping des gradients pour la stabilite
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1

        return {"loss": loss.item(), "q_value": q_value.item()}

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def on_episode_end(self, total_reward, episode_length):
        """Decroissance epsilon apres chaque episode."""
        super().on_episode_end(total_reward, episode_length)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_training_mode(self, training: bool):
        self._training = training
        if training:
            self.q_net.train()
        else:
            self.q_net.eval()

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "q_net": self.q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)


# ============================================================================
# 2. DOUBLE DEEP Q-LEARNING (DDQN)
# ============================================================================

class DoubleDeepQLearning(DeepQLearning):
    """
    Double Deep Q-Learning — Ajoute un reseau cible.

    Probleme du DQN : la cible max_a' Q(s',a';theta) utilise le meme
    reseau pour SELECTIONNER et EVALUER l'action, ce qui surestime
    systematiquement les Q-valeurs (biais de maximisation).

    Solution (van Hasselt et al., 2016) :
    - Reseau en ligne theta : selectionne l'action
          a* = argmax_a' Q(s', a'; theta)
    - Reseau cible theta^- : evalue la cible
          y = r + gamma * Q(s', a*; theta^-)
    - theta^- est copie depuis theta toutes les C etapes

    Cible corrigee :
        y = r + gamma * Q(s', argmax_a' Q(s', a'; theta) ; theta^-)

    Le decouplage selection/evaluation reduit la surestimation.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
    ):
        """
        Args:
            ... (memes que DeepQLearning)
            target_update_freq: Frequence de copie theta -> theta^- (en steps)
        """
        super().__init__(
            state_dim, n_actions, hidden_dims, lr, gamma,
            epsilon_start, epsilon_end, epsilon_decay,
        )
        self.name = "DoubleDeepQLearning"

        # Reseau cible (copie du reseau en ligne)
        self.target_net = MLP(state_dim, n_actions, self.hidden_dims).to(self.device)
        self.sync_target()
        self.target_update_freq = target_update_freq

    def sync_target(self):
        """Copie les poids theta -> theta^-."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        Mise a jour avec Double Q-Learning.

        1. Selection : a* = argmax_a' Q(s', a'; theta)      [reseau en ligne]
        2. Evaluation : y = r + gamma * Q(s', a*; theta^-)   [reseau cible]
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Q(s, a; theta)
        q_value = self.q_net(state_t)[0, action]

        with torch.no_grad():
            if done:
                target = torch.tensor(
                    reward, device=self.device, dtype=torch.float32
                )
            else:
                # 1. Selection par le reseau en ligne
                next_q_online = self.q_net(next_state_t)

                available_next = kwargs.get("available_actions_next")
                if available_next:
                    mask = torch.full(
                        (self.n_actions,), float("-inf"), device=self.device
                    )
                    for a in available_next:
                        mask[a] = 0.0
                    next_q_online = next_q_online + mask

                best_action = next_q_online.argmax(dim=1)

                # 2. Evaluation par le reseau cible
                next_q_target = self.target_net(next_state_t)
                target = reward + self.gamma * next_q_target[0, best_action].squeeze()

        # Assurer la meme forme pour la loss
        loss = self.loss_fn(q_value.unsqueeze(0), target.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1

        # Synchronisation periodique du reseau cible
        if self.training_steps % self.target_update_freq == 0:
            self.sync_target()

        return {"loss": loss.item(), "q_value": q_value.item()}

    def set_training_mode(self, training: bool):
        super().set_training_mode(training)
        if training:
            self.target_net.train()
        else:
            self.target_net.eval()

    def save(self, path):
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_update_freq": self.target_update_freq,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)


# ============================================================================
# 3. DDQN + EXPERIENCE REPLAY
# ============================================================================

class DDQNWithExperienceReplay(DoubleDeepQLearning):
    """
    Double DQN avec Experience Replay (Mnih et al., 2015 + van Hasselt 2016).

    L'Experience Replay casse les correlations temporelles entre
    transitions consecutives en :
    1. Stockant chaque transition (s, a, r, s', done) dans un buffer
    2. Echantillonnant des mini-batches uniformement pour la mise a jour

    Avantages :
    - Reutilise chaque experience ~buffer_size/batch_size fois
    - Reduit la variance des mises a jour
    - Lisse la distribution d'apprentissage

    Difference avec les variants precedentes :
    - DQN/DDQN : learn() fait une MAJ sur la transition courante
    - DDQN+ER : learn() stocke puis MAJ sur un batch echantillonne

    Le masque des actions valides du next_state est stocke dans le buffer
    avec chaque transition et applique avant l'argmax de selection, ce qui
    evite de selectionner des actions invalides comme meilleure action.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        buffer_size: int = 10000,
        batch_size: int = 32,
        min_buffer_size: int = 100,
    ):
        """
        Args:
            ... (memes que DoubleDeepQLearning)
            buffer_size: Capacite du buffer de rejeu
            batch_size: Taille du mini-batch
            min_buffer_size: Transitions min avant de commencer l'apprentissage
        """
        super().__init__(
            state_dim, n_actions, hidden_dims, lr, gamma,
            epsilon_start, epsilon_end, epsilon_decay, target_update_freq,
        )
        self.name = "DDQN_ER"

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.buffer_size = buffer_size

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        1. Stocke la transition dans le buffer (avec masque d'actions valides)
        2. Si assez de transitions : echantillonne un batch et MAJ
        """
        # Construire le masque des actions valides du next_state
        available_next = kwargs.get("available_actions_next")
        if done or not available_next:
            next_mask = np.ones(self.n_actions, dtype=np.float32)
        else:
            next_mask = np.zeros(self.n_actions, dtype=np.float32)
            for a in available_next:
                next_mask[a] = 1.0

        # Stocker avec le masque
        self.buffer.push(state, action, reward, next_state, float(done), next_mask)

        # Attendre que le buffer soit assez rempli
        if len(self.buffer) < self.min_buffer_size:
            return None

        # Echantillonner un mini-batch
        states, actions_b, rewards_b, next_states, dones_b, next_masks_b = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions_b).to(self.device)
        rewards_t = torch.FloatTensor(rewards_b).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones_b).to(self.device)

        # Q(s, a; theta) pour le batch
        q_values = self.q_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Double Q-Learning targets (vectorise)
        with torch.no_grad():
            # Selection par reseau en ligne avec masquage des actions invalides
            next_q_online = self.q_net(next_states_t)
            if next_masks_b is not None:
                masks_t = torch.FloatTensor(next_masks_b).to(self.device)
                next_q_online = torch.where(
                    masks_t.bool(), next_q_online,
                    torch.full_like(next_q_online, float('-inf'))
                )
            best_actions = next_q_online.argmax(dim=1)

            # Evaluation par reseau cible
            next_q_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_freq == 0:
            self.sync_target()

        return {"loss": loss.item(), "q_value": q_values.mean().item()}

    def save(self, path):
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_update_freq": self.target_update_freq,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "min_buffer_size": self.min_buffer_size,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )


# ============================================================================
# 4. DDQN + PRIORITIZED EXPERIENCE REPLAY
# ============================================================================

class DDQNWithPrioritizedExperienceReplay(DoubleDeepQLearning):
    """
    Double DQN avec Prioritized Experience Replay (Schaul et al., 2016).

    Motivation : dans un buffer uniforme, toutes les transitions ont la
    meme probabilite d'etre echantillonnees. Or certaines sont plus
    informatives (fort TD-error = surprise importante).

    Echantillonnage prioritaire :
        P(i) = p_i^alpha / sum_k p_k^alpha
        ou p_i = |delta_i| + epsilon

    Le biais introduit est corrige par Importance Sampling :
        w_i = (N * P(i))^{-beta}  (normalise par max)

    beta augmente lineairement de beta_start a 1.0 pour une correction
    de plus en plus exacte au fil de l'entrainement.

    Complexite : O(log n) par echantillonnage grace au SumTree.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims=None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        buffer_size: int = 10000,
        batch_size: int = 32,
        min_buffer_size: int = 100,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
    ):
        """
        Args:
            ... (memes que DoubleDeepQLearning)
            buffer_size: Capacite du buffer
            batch_size: Taille du mini-batch
            min_buffer_size: Transitions min avant apprentissage
            alpha: Exposant de priorite (0=uniforme, 1=full priority)
            beta_start: Beta initial pour IS
            beta_end: Beta final (1.0 = correction complete)
            beta_frames: Nombre de steps pour l'annealing de beta
        """
        super().__init__(
            state_dim, n_actions, hidden_dims, lr, gamma,
            epsilon_start, epsilon_end, epsilon_decay, target_update_freq,
        )
        self.name = "DDQN_PER"

        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha)
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.buffer_size = buffer_size

        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames

    def learn(self, state, action, reward, next_state, done, **kwargs):
        """
        1. Stocke la transition avec priorite maximale (et masque d'actions valides)
        2. Echantillonne selon les priorites
        3. MAJ avec poids IS
        4. Met a jour les priorites avec les nouveaux TD-errors
        """
        # Construire le masque des actions valides du next_state
        available_next = kwargs.get("available_actions_next")
        if done or not available_next:
            next_mask = np.ones(self.n_actions, dtype=np.float32)
        else:
            next_mask = np.zeros(self.n_actions, dtype=np.float32)
            for a in available_next:
                next_mask[a] = 1.0

        # Stocker avec le masque
        self.buffer.push(state, action, reward, next_state, float(done), next_mask)

        if len(self.buffer) < self.min_buffer_size:
            return None

        # Annealing de beta : beta_start -> beta_end
        fraction = min(1.0, self.training_steps / max(1, self.beta_frames))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

        # Echantillonner avec priorites
        (
            states, actions_b, rewards_b, next_states, dones_b,
            indices, is_weights, next_masks_b,
        ) = self.buffer.sample(self.batch_size, self.beta)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions_b).to(self.device)
        rewards_t = torch.FloatTensor(rewards_b).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones_b).to(self.device)
        weights_t = torch.FloatTensor(is_weights).to(self.device)

        # Q(s, a; theta)
        q_values = self.q_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Double Q-Learning targets avec masquage des actions invalides
        with torch.no_grad():
            next_q_online = self.q_net(next_states_t)
            if next_masks_b is not None:
                masks_t = torch.FloatTensor(next_masks_b).to(self.device)
                next_q_online = torch.where(
                    masks_t.bool(), next_q_online,
                    torch.full_like(next_q_online, float('-inf'))
                )
            best_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # TD-errors pour mise a jour des priorites
        td_errors = (q_values - targets).detach()

        # Perte ponderee par IS (corrige le biais de l'echantillonnage)
        loss = (weights_t * (q_values - targets).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Mettre a jour les priorites dans le buffer
        self.buffer.update_priorities(indices, td_errors.cpu().numpy())

        self.training_steps += 1

        if self.training_steps % self.target_update_freq == 0:
            self.sync_target()

        return {"loss": loss.item(), "q_value": q_values.mean().item()}

    def save(self, path):
        torch.save(
            {
                "class": self.__class__.__name__,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "hidden_dims": self.hidden_dims,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "target_update_freq": self.target_update_freq,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "min_buffer_size": self.min_buffer_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "beta_frames": self.beta_frames,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "episodes_played": self.episodes_played,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.beta = ckpt.get("beta", self.beta_start)
        self.training_steps = ckpt.get("training_steps", 0)
        self.episodes_played = ckpt.get("episodes_played", 0)
