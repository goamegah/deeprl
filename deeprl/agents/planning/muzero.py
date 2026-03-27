"""
MuZero - Modèle appris de l'environnement.

MuZero (Schrittwieser et al., 2020) étend AlphaZero en apprenant
un MODÈLE de l'environnement au lieu de le simuler directement.

Trois réseaux:
1. Representation h: observation → état latent
2. Dynamics g: (état_latent, action) → (nouvel_état, reward)
3. Prediction f: état_latent → (policy, value)

Avantages:
- Fonctionne sans accès au simulateur
- Peut apprendre dans des environnements complexes
- Combine planning et model-based RL

MCTS dans l'espace latent:
- Utilise le modèle Dynamics au lieu du vrai environnement
- Plus rapide et plus flexible
- Le modèle apprend ce qui est important pour la décision

Référence:
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import math
from collections import deque

from deeprl.agents.base import Agent


class RepresentationNetwork(nn.Module):
    """
    Réseau de représentation h: observation → état latent.
    
    Encode l'observation en un état latent compact
    utilisé par les autres réseaux.
    """
    
    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        super().__init__()
        
        layers = []
        prev_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: (batch, observation_dim)
        Returns:
            latent_state: (batch, latent_dim)
        """
        return self.network(observation)


class DynamicsNetwork(nn.Module):
    """
    Réseau de dynamique g: (état_latent, action) → (nouvel_état, reward).
    
    Prédit comment l'état évolue après une action
    et quelle récompense on obtient.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [128]
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        
        # Encoder l'action en one-hot puis concaténer avec l'état
        input_dim = latent_dim + n_actions
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Sorties
        self.next_state_head = nn.Linear(prev_dim, latent_dim)
        self.reward_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        latent_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: (batch, latent_dim)
            action: (batch,) indices des actions
        
        Returns:
            next_latent_state: (batch, latent_dim)
            reward: (batch, 1)
        """
        batch_size = latent_state.size(0)
        
        # One-hot encoding de l'action
        action_onehot = torch.zeros(
            batch_size, self.n_actions,
            device=latent_state.device
        )
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        # Concaténer
        x = torch.cat([latent_state, action_onehot], dim=-1)
        
        trunk_out = self.trunk(x)
        next_state = self.next_state_head(trunk_out)
        reward = self.reward_head(trunk_out)
        
        return next_state, reward


class PredictionNetwork(nn.Module):
    """
    Réseau de prédiction f: état_latent → (policy, value).
    
    Prédit la politique optimale et la valeur
    à partir de l'état latent.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [128]
    ):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        self.policy_head = nn.Linear(prev_dim, n_actions)
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    
    def forward(
        self,
        latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: (batch, latent_dim)
        
        Returns:
            policy_logits: (batch, n_actions)
            value: (batch, 1)
        """
        trunk_out = self.trunk(latent_state)
        policy = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)
        return policy, value


@dataclass
class MuZeroNode:
    """Nœud MCTS pour MuZero (dans l'espace latent)."""
    
    latent_state: torch.Tensor
    parent: Optional["MuZeroNode"] = None
    action: Optional[int] = None
    
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    reward: float = 0.0  # Récompense prédite par Dynamics
    
    children: Dict[int, "MuZeroNode"] = field(default_factory=dict)
    is_expanded: bool = False
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration
    
    def select_child(self, c_puct: float) -> "MuZeroNode":
        return max(
            self.children.values(),
            key=lambda n: n.ucb_score(c_puct, self.visit_count)
        )


class MuZeroAgent(Agent):
    """
    Agent MuZero.
    
    Apprend un modèle de l'environnement et l'utilise
    pour planifier avec MCTS dans l'espace latent.
    
    Caractéristiques:
    - Ne nécessite pas de simulateur pour la planification
    - Apprend ce qui est pertinent pour la décision
    - Combine les avantages de model-free et model-based
    
    Exemple:
        >>> agent = MuZeroAgent(state_dim=27, n_actions=9)
        >>> action = agent.act(state)  # Pas besoin d'env pour MCTS!
        >>> # Entraînement
        >>> agent.store_transition(state, action, reward, next_state, done)
        >>> if len(agent.replay_buffer) > 100:
        >>>     agent.train_step()
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        # Architecture
        latent_dim: int = 64,
        hidden_dims: List[int] = [128, 128],
        # MCTS
        n_simulations: int = 50,
        c_puct: float = 1.25,
        # Entraînement
        lr: float = 1e-3,
        gamma: float = 0.99,
        unroll_steps: int = 5,  # Nombre de steps pour le unrolling
        # Buffer
        buffer_size: int = 10000,
        batch_size: int = 32,
        # Autres
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent MuZero.
        
        Args:
            state_dim: Dimension de l'observation
            n_actions: Nombre d'actions
            latent_dim: Dimension de l'espace latent
            hidden_dims: Architecture des réseaux
            n_simulations: Simulations MCTS par décision
            c_puct: Constante d'exploration
            lr: Learning rate
            gamma: Facteur d'actualisation
            unroll_steps: Nombre de steps pour l'entraînement
            buffer_size: Taille du replay buffer
            batch_size: Taille des batches
            device: "cpu" ou "cuda"
            seed: Graine aléatoire
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="MuZero",
            device=device
        )
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.lr = lr
        self.gamma = gamma
        self.unroll_steps = unroll_steps
        self.batch_size = batch_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Les trois réseaux
        self.representation = RepresentationNetwork(
            observation_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.dynamics = DynamicsNetwork(
            latent_dim=latent_dim,
            n_actions=n_actions,
            hidden_dims=[hidden_dims[0]] if hidden_dims else [128]
        ).to(self.device)
        
        self.prediction = PredictionNetwork(
            latent_dim=latent_dim,
            n_actions=n_actions,
            hidden_dims=[hidden_dims[0]] if hidden_dims else [128]
        ).to(self.device)
        
        # Optimiseur pour tous les réseaux
        self.optimizer = optim.Adam(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            lr=lr
        )
        
        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=buffer_size)
        
        # Trajectoire courante
        self._current_trajectory: List[Dict] = []
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action via MCTS dans l'espace latent.
        
        Note: Contrairement à AlphaZero, pas besoin de l'environnement!
        """
        # Encoder l'observation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent = self.representation(state_tensor)
        
        # Créer la racine
        root = MuZeroNode(latent_state=latent)
        
        # Obtenir la policy et value initiales
        with torch.no_grad():
            policy_logits, value = self.prediction(latent)
            policy = F.softmax(policy_logits, dim=-1)[0]
        
        # Expand la racine
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        
        for action in available_actions:
            child = MuZeroNode(
                latent_state=None,  # Sera calculé lors de la visite
                parent=root,
                action=action,
                prior=policy[action].item()
            )
            root.children[action] = child
        root.is_expanded = True
        
        # Simulations MCTS
        for _ in range(self.n_simulations):
            node = root
            path = [node]
            
            # Selection
            while node.is_expanded and len(node.children) > 0:
                node = node.select_child(self.c_puct)
                path.append(node)
            
            # Expansion avec le modèle Dynamics
            if not node.is_expanded and node.latent_state is None:
                # Calculer l'état latent avec Dynamics
                parent_latent = node.parent.latent_state
                action_tensor = torch.LongTensor([node.action]).to(self.device)
                
                with torch.no_grad():
                    next_latent, reward = self.dynamics(parent_latent, action_tensor)
                
                node.latent_state = next_latent
                node.reward = reward.item()
                
                # Obtenir policy et value
                with torch.no_grad():
                    policy_logits, value = self.prediction(next_latent)
                    policy = F.softmax(policy_logits, dim=-1)[0]
                
                # Créer les enfants
                for a in available_actions:
                    child = MuZeroNode(
                        latent_state=None,
                        parent=node,
                        action=a,
                        prior=policy[a].item()
                    )
                    node.children[a] = child
                node.is_expanded = True
                
                value = value.item()
            else:
                # Utiliser la value du réseau
                with torch.no_grad():
                    _, value_tensor = self.prediction(node.latent_state)
                value = value_tensor.item()
            
            # Backpropagation
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = n.reward + self.gamma * value
        
        # Choisir l'action
        if training:
            # Échantillonnage proportionnel aux visites
            visits = np.array([
                root.children[a].visit_count if a in root.children else 0
                for a in range(self.n_actions)
            ])
            if visits.sum() > 0:
                probs = visits / visits.sum()
                action = self.rng.choice(self.n_actions, p=probs)
            else:
                action = self.rng.choice(available_actions)
        else:
            # Argmax
            action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
        
        # Stocker la politique MCTS pour l'entraînement
        policy_target = np.zeros(self.n_actions, dtype=np.float32)
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits > 0:
            for a, child in root.children.items():
                policy_target[a] = child.visit_count / total_visits
        
        self._last_policy = policy_target
        self._last_value = root.value
        
        return int(action)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        policy: Optional[np.ndarray] = None,
        value: Optional[float] = None
    ):
        """
        Stocke une transition dans le buffer.
        
        Args:
            state: État courant
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Épisode terminé
            policy: Politique MCTS (pour entraînement)
            value: Valeur MCTS (pour entraînement)
        """
        if policy is None:
            policy = getattr(self, '_last_policy', np.ones(self.n_actions) / self.n_actions)
        if value is None:
            value = getattr(self, '_last_value', 0.0)
        
        self._current_trajectory.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'policy': policy.copy(),
            'value': value
        })
        
        if done:
            # Calculer les valeurs cibles avec les récompenses réelles
            trajectory = self._current_trajectory
            n = len(trajectory)
            
            for i, trans in enumerate(trajectory):
                # Valeur cible = récompenses futures actualisées
                target_value = 0.0
                for j in range(i, min(i + self.unroll_steps, n)):
                    target_value += (self.gamma ** (j - i)) * trajectory[j]['reward']
                
                trans['target_value'] = target_value
            
            # Ajouter au replay buffer
            self.replay_buffer.extend(trajectory)
            self._current_trajectory = []
    
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ) -> Optional[Dict[str, float]]:
        """Interface standard."""
        policy = kwargs.get('policy', None)
        value = kwargs.get('value', None)
        
        self.store_transition(state, action, reward, next_state, done, policy, value)
        
        if done and len(self.replay_buffer) >= self.batch_size:
            return self.train_step()
        
        return None
    
    def train_step(self) -> Dict[str, float]:
        """
        Effectue un step d'entraînement.
        
        Entraîne les trois réseaux sur un batch du replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Échantillonner un batch
        indices = self.rng.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Préparer les tenseurs (utiliser np.array pour éviter le warning)
        states = torch.FloatTensor(np.array([t['state'] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t['action'] for t in batch])).to(self.device)
        target_policies = torch.FloatTensor(np.array([t['policy'] for t in batch])).to(self.device)
        target_values = torch.FloatTensor(np.array([[t.get('target_value', t['value'])] for t in batch])).to(self.device)
        target_rewards = torch.FloatTensor(np.array([[t['reward']] for t in batch])).to(self.device)
        
        # Forward
        latent = self.representation(states)
        policy_logits, values = self.prediction(latent)
        
        # Dynamics pour prédire la récompense
        next_latent, pred_rewards = self.dynamics(latent, actions)
        
        # Losses
        # 1. Policy loss (cross-entropy)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(target_policies * log_probs) / self.batch_size
        
        # 2. Value loss (MSE)
        value_loss = F.mse_loss(values, target_values)
        
        # 3. Reward loss (MSE)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        
        # Total loss
        loss = policy_loss + value_loss + reward_loss
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.training_steps += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item(),
        }
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        torch.save({
            'representation': self.representation.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'prediction': self.prediction.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'config': self.get_config(),
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.representation.load_state_dict(checkpoint['representation'])
        self.dynamics.load_state_dict(checkpoint['dynamics'])
        self.prediction.load_state_dict(checkpoint['prediction'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint.get('training_steps', 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'type': 'MuZero',
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'n_simulations': self.n_simulations,
            'c_puct': self.c_puct,
            'lr': self.lr,
            'gamma': self.gamma,
            'unroll_steps': self.unroll_steps,
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test de MuZero ===\n")
    
    from deeprl.envs.tictactoe import TicTacToeVsRandom
    
    env = TicTacToeVsRandom()
    agent = MuZeroAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        latent_dim=32,
        hidden_dims=[64],
        n_simulations=20
    )
    
    print(f"Agent: {agent.name}")
    print(f"Config: {agent.get_config()}")
    
    # Tester l'action
    print("\n1. Test action:")
    state = env.reset()
    action = agent.act(state)
    print(f"   Action choisie: {action}")
    
    # Jouer quelques parties pour remplir le buffer
    print("\n2. Collecte de données (5 parties):")
    for game in range(5):
        state = env.reset()
        while not env.is_game_over:
            available = env.get_available_actions()
            action = agent.act(state, available)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
    
    print(f"   Buffer size: {len(agent.replay_buffer)}")
    
    # Entraînement
    print("\n3. Entraînement:")
    for i in range(5):
        metrics = agent.train_step()
        if metrics:
            print(f"   Step {i+1}: policy_loss={metrics['policy_loss']:.4f}, "
                  f"value_loss={metrics['value_loss']:.4f}, "
                  f"reward_loss={metrics['reward_loss']:.4f}")
    
    print("\n[OK] Tests passes!")
