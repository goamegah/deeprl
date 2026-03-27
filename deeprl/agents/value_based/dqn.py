"""
Deep Q-Network (DQN) - Le premier algorithme de Deep RL à succès.

DQN combine Q-Learning avec des réseaux de neurones profonds.
C'est l'algorithme qui a permis de jouer aux jeux Atari au niveau humain.

Innovations clés:
1. Experience Replay: casse la corrélation temporelle
2. Target Network: stabilise l'entraînement

Équation de perte:
    L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))²]

où Q_target est une copie "gelée" du réseau mise à jour périodiquement.

Variantes implémentées ici:
- DeepQLearning: DQN sans replay (apprentissage online)
- DoubleDeepQLearning: Double DQN sans replay
- DoubleDeepQLearningWithER: Double DQN + Experience Replay
- DoubleDeepQLearningWithPER: Double DQN + Prioritized Experience Replay
- Dueling: architecture Dueling (combinable avec les autres)

Référence:
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- "Human-level control through deep RL" (Mnih et al., 2015)
"""

import numpy as np
import torch
import torch.optim as optim
from typing import List, Optional, Dict, Any

from deeprl.agents.base import Agent
from deeprl.networks.mlp import MLP, DuelingMLP
from deeprl.memory.replay_buffer import ReplayBuffer
from deeprl.memory.prioritized_buffer import PrioritizedReplayBuffer


class DQNAgent(Agent):
    """
    Agent Deep Q-Network avec toutes les améliorations.
    
    Caractéristiques:
    - Réseau de neurones pour approximer Q(s, a)
    - Experience Replay pour la stabilité
    - Target Network pour réduire l'instabilité
    - Support de Double DQN
    - Support de Dueling DQN
    - Support de Prioritized Experience Replay
    
    Exemple d'utilisation:
        >>> agent = DQNAgent(state_dim=4, n_actions=2)
        >>> action = agent.act(state)
        >>> agent.learn(state, action, reward, next_state, done)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        # Architecture
        hidden_dims: List[int] = [64, 64],
        dueling: bool = False,
        # Hyperparamètres d'apprentissage
        lr: float = 1e-3,
        gamma: float = 0.99,
        # Exploration
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        # Experience Replay
        use_replay: bool = True,
        buffer_size: int = 10000,
        batch_size: int = 64,
        min_buffer_size: int = 1000,
        # Prioritized Experience Replay
        prioritized: bool = False,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        # Target Network
        target_update_freq: int = 100,
        soft_update: bool = False,
        tau: float = 0.005,
        # Double DQN
        double_dqn: bool = True,
        # Autres
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent DQN.
        
        Args:
            state_dim: Dimension de l'espace d'états
            n_actions: Nombre d'actions possibles
            hidden_dims: Architecture du réseau (ex: [64, 64])
            dueling: Utiliser l'architecture Dueling DQN
            lr: Taux d'apprentissage
            gamma: Facteur d'actualisation
            epsilon_start/end/decay: Paramètres d'exploration ε-greedy
            use_replay: Si True, utilise Experience Replay (recommandé)
            buffer_size: Taille du replay buffer
            batch_size: Taille des mini-batches
            min_buffer_size: Taille minimale avant apprentissage
            prioritized: Utiliser Prioritized Experience Replay (requiert use_replay=True)
            alpha: Exposant de priorité (si prioritized=True)
            beta_start: Beta initial pour importance sampling
            target_update_freq: Fréquence de mise à jour du target network
            soft_update: Si True, mise à jour douce (Polyak averaging)
            tau: Coefficient pour soft update
            double_dqn: Utiliser Double DQN
            device: "cpu" ou "cuda"
            seed: Graine aléatoire
        """
        # Construire le nom selon les options (style prof)
        name_parts = ["DQN"]
        if double_dqn:
            name_parts = ["Double"] + name_parts
        if dueling:
            name_parts.append("Dueling")
        if use_replay:
            if prioritized:
                name_parts.append("PER")
            else:
                name_parts.append("ER")
        
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name=" ".join(name_parts),
            device=device
        )
        
        # Sauvegarder les hyperparamètres
        self.hidden_dims = hidden_dims
        self.dueling = dueling
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_replay = use_replay
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.soft_update = soft_update
        self.tau = tau
        self.double_dqn = double_dqn
        self.prioritized = prioritized
        
        # Validation: PER requiert Experience Replay
        if prioritized and not use_replay:
            raise ValueError("Prioritized Experience Replay requires use_replay=True")
        
        # Générateur aléatoire
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Créer les réseaux
        if dueling:
            self.q_network = DuelingMLP(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.target_network = DuelingMLP(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dims=hidden_dims
            ).to(self.device)
        else:
            self.q_network = MLP(
                state_dim=state_dim,
                output_dim=n_actions,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.target_network = MLP(
                state_dim=state_dim,
                output_dim=n_actions,
                hidden_dims=hidden_dims
            ).to(self.device)
        
        # Copier les poids vers le target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Mode évaluation
        
        # Optimiseur
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay Buffer (optionnel)
        if use_replay:
            if prioritized:
                self.buffer = PrioritizedReplayBuffer(
                    capacity=buffer_size,
                    alpha=alpha,
                    beta_start=beta_start
                )
            else:
                self.buffer = ReplayBuffer(capacity=buffer_size)
        else:
            self.buffer = None
        
        # Compteurs
        self._training = True
        self._update_counter = 0
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action selon la politique ε-greedy.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: Mode entraînement ou évaluation
        
        Returns:
            Action choisie
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        
        # Exploration vs Exploitation
        use_training = training and self._training
        
        if use_training and self.rng.random() < self.epsilon:
            # Exploration: action aléatoire
            return int(self.rng.choice(available_actions))
        
        # Exploitation: meilleure action selon Q
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Masquer les actions non disponibles
        masked_q = np.full(self.n_actions, -np.inf)
        for a in available_actions:
            masked_q[a] = q_values[a]
        
        return int(np.argmax(masked_q))
    
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ) -> Optional[Dict[str, float]]:
        """
        Effectue une mise à jour (avec ou sans Experience Replay).
        
        Args:
            state, action, reward, next_state, done: Transition
        
        Returns:
            Dictionnaire avec les métriques (loss, q_value, etc.)
        """
        if self.use_replay:
            # Mode Experience Replay: stocker et échantillonner
            self.buffer.push(state, action, reward, next_state, done)
            
            # Vérifier si on peut apprendre
            if not self.buffer.is_ready(self.min_buffer_size):
                return None
            
            # Échantillonner et apprendre
            loss_info = self._update()
        else:
            # Mode online: apprendre directement sur la transition courante
            loss_info = self._update_online(state, action, reward, next_state, done)
        
        # Mettre à jour le target network
        self._update_counter += 1
        if self._update_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        self.training_steps += 1
        
        return loss_info
    
    def _update(self) -> Dict[str, float]:
        """
        Effectue une mise à jour du réseau Q.
        
        Returns:
            Métriques de la mise à jour
        """
        # Échantillonner un batch
        if self.prioritized:
            batch, weights, indices = self.buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        states, actions, rewards, next_states, dones = batch
        
        # Convertir en tenseurs
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculer Q(s, a)
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculer la cible
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: sélectionner avec q_network, évaluer avec target
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states)
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # DQN standard
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calculer la TD error
        td_errors = targets - q_values
        
        # Loss pondérée (pour PER)
        loss = (weights * td_errors.pow(2)).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (stabilité)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        # Mettre à jour les priorités (PER)
        if self.prioritized and indices is not None:
            new_priorities = td_errors.abs().detach().cpu().numpy()
            self.buffer.update_priorities(indices, new_priorities)
        
        return {
            "loss": loss.item(),
            "q_value": q_values.mean().item(),
            "td_error": td_errors.abs().mean().item(),
        }
    
    def _update_online(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Mise à jour online (sans Experience Replay).
        
        Apprend directement sur la transition courante.
        Plus instable mais correspond à DeepQLearning/DoubleDeepQLearning
        sans buffer, comme demandé.
        
        Args:
            state, action, reward, next_state, done: Transition courante
        
        Returns:
            Métriques de la mise à jour
        """
        # Convertir en tenseurs (batch de taille 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        reward_t = torch.FloatTensor([reward]).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done_t = torch.FloatTensor([float(done)]).to(self.device)
        
        # Calculer Q(s, a)
        q_values = self.q_network(state_t)
        q_value = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)
        
        # Calculer la cible
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: sélectionner avec q_network, évaluer avec target
                next_action = self.q_network(next_state_t).argmax(dim=1)
                next_q_value = self.target_network(next_state_t)
                next_q_value = next_q_value.gather(1, next_action.unsqueeze(1)).squeeze(1)
            else:
                # DQN standard
                next_q_value = self.target_network(next_state_t).max(dim=1)[0]
            
            target = reward_t + self.gamma * next_q_value * (1 - done_t)
        
        # Calculer la TD error et la loss
        td_error = target - q_value
        loss = td_error.pow(2).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (stabilité)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "q_value": q_value.item(),
            "td_error": abs(td_error.item()),
        }
    
    def _update_target_network(self) -> None:
        """Met à jour le target network."""
        if self.soft_update:
            # Soft update (Polyak averaging)
            for target_param, param in zip(
                self.target_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        else:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def on_episode_end(self, total_reward: float, episode_length: int) -> None:
        """Décroît epsilon à la fin de chaque épisode."""
        super().on_episode_end(total_reward, episode_length)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def set_training_mode(self, training: bool) -> None:
        """Active ou désactive le mode entraînement."""
        self._training = training
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Retourne les valeurs Q pour un état."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
            "config": self.get_config(),
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint.get("training_steps", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'agent."""
        config = super().get_config()
        config.update({
            "type": self.name,
            "hidden_dims": self.hidden_dims,
            "dueling": self.dueling,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "double_dqn": self.double_dqn,
            "use_replay": self.use_replay,
            "prioritized": self.prioritized,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        })
        return config
    
    def __repr__(self) -> str:
        if self.use_replay:
            buffer_info = f"buffer={len(self.buffer)}"
        else:
            buffer_info = "online"
        return (
            f"{self.name}(state_dim={self.state_dim}, n_actions={self.n_actions}, "
            f"ε={self.epsilon:.3f}, {buffer_info})"
        )


# Test rapide
if __name__ == "__main__":
    print("=== Test de DQN - Toutes variantes du prof ===\n")
    
    # Configuration
    state_dim = 25  # GridWorld 5x5
    n_actions = 4
    
    # Test 1: DeepQLearning (sans Double, sans ER)
    print("1. DeepQLearning (DQN sans replay):")
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        double_dqn=False,
        use_replay=False
    )
    print(f"   {agent}")
    
    # Test 2: DoubleDeepQLearning (Double, sans ER)
    print("\n2. DoubleDeepQLearning (Double DQN sans replay):")
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        double_dqn=True,
        use_replay=False
    )
    print(f"   {agent}")
    
    # Test 3: DoubleDeepQLearningWithER
    print("\n3. DoubleDeepQLearningWithER (Double DQN + ER):")
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        double_dqn=True,
        use_replay=True,
        prioritized=False
    )
    print(f"   {agent}")
    
    # Test 4: DoubleDeepQLearningWithPER
    print("\n4. DoubleDeepQLearningWithPER (Double DQN + PER):")
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        double_dqn=True,
        use_replay=True,
        prioritized=True
    )
    print(f"   {agent}")
    
    # Test 5: Simulation online (sans ER)
    print("\n5. Test apprentissage online (sans ER):")
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        double_dqn=True,
        use_replay=False
    )
    
    for i in range(20):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.act(state)
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        done = (i == 19)
        
        result = agent.learn(state, action, reward, next_state, done)
        if i % 5 == 0:
            print(f"   Step {i}: loss={result['loss']:.4f}, Q={result['q_value']:.4f}")
    
    print(f"   Training steps: {agent.training_steps}")
    
    # Test 6: Simulation avec ER
    print("\n6. Test apprentissage avec ER:")
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        double_dqn=True,
        use_replay=True,
        min_buffer_size=50,
        batch_size=16
    )
    
    for i in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.act(state)
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        done = (i == 99)
        
        result = agent.learn(state, action, reward, next_state, done)
        if result is not None and i % 20 == 0:
            print(f"   Step {i}: loss={result['loss']:.4f}, Q={result['q_value']:.4f}")
    
    print(f"   Buffer size: {len(agent.buffer)}")
    print(f"   Training steps: {agent.training_steps}")
    
    print("\n[OK] Tests passes!")
