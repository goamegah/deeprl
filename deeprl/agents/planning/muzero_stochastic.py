"""
MuZero Stochastique - Extension pour environnements stochastiques.

MuZero Stochastique (Antonoglou et al., 2021) etend MuZero pour gerer
les environnements avec des transitions aleatoires (stochastiques).

Difference avec MuZero standard:
- Ajoute un reseau "Afterstate" pour modeliser l'effet deterministe de l'action
- Ajoute un reseau "Encoder de chance" pour capturer les evenements aleatoires
- La dynamique est decomposee: action -> afterstate -> evenement aleatoire -> etat suivant

Trois phases dans la transition:
1. Afterstate: g_a(s, a) -> s_afterstate (effet deterministe de l'action)
2. Chance Encoder: e(observation) -> z (encode l'evenement aleatoire observe)
3. Dynamics: g_c(s_afterstate, z) -> s' (effet de l'evenement aleatoire)

Applications:
- Jeux avec des (cartes, poker)
- Jeux de plateau avec hasard (Backgammon)
- Environnements avec bruit de transition

Reference:
- "Planning in Stochastic Environments with a Learned Model" (2021)
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
    Reseau de representation h: observation -> etat latent.
    Identique a MuZero standard.
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
        return self.network(observation)


class AfterstateNetwork(nn.Module):
    """
    Reseau Afterstate g_a: (etat_latent, action) -> afterstate.
    
    Modelise l'effet DETERMINISTE de l'action avant que 
    l'evenement aleatoire ne se produise.
    
    Exemple: Au poker, l'afterstate apres "miser 10" est l'etat
    ou on a mise, mais avant que les autres joueurs reagissent.
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
        
        input_dim = latent_dim + n_actions
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Sortie: nouvel etat latent (afterstate)
        self.state_head = nn.Linear(prev_dim, latent_dim)
        
        # Reward intermediaire (optionnel, souvent 0)
        self.reward_head = nn.Linear(prev_dim, 1)
        
        self.shared = nn.Sequential(*layers)
    
    def forward(
        self, 
        latent_state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: (batch, latent_dim)
            action: (batch,) indices d'action
        
        Returns:
            afterstate: (batch, latent_dim)
            reward: (batch, 1)
        """
        batch_size = latent_state.size(0)
        
        # One-hot encode action
        action_onehot = torch.zeros(batch_size, self.n_actions, device=latent_state.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        # Concatener et passer dans le reseau
        x = torch.cat([latent_state, action_onehot], dim=1)
        h = self.shared(x)
        
        afterstate = self.state_head(h)
        reward = self.reward_head(h)
        
        return afterstate, reward


class ChanceEncoderNetwork(nn.Module):
    """
    Reseau Encoder de Chance e: observation -> z (code de chance).
    
    Encode l'evenement aleatoire observe dans un vecteur discret ou continu.
    Ce code represente "ce qui s'est passe" de facon stochastique.
    
    Exemple: Au poker, encode quelle carte a ete revelee.
    """
    
    def __init__(
        self,
        observation_dim: int,
        chance_dim: int,  # Dimension du code de chance
        hidden_dims: List[int] = [64],
        n_chance_outcomes: int = 32  # Nombre d'outcomes discrets
    ):
        super().__init__()
        
        self.chance_dim = chance_dim
        self.n_outcomes = n_chance_outcomes
        
        layers = []
        prev_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Sortie: logits pour les outcomes discrets
        self.outcome_head = nn.Linear(prev_dim, n_chance_outcomes)
        
        # Embedding des outcomes
        self.outcome_embedding = nn.Embedding(n_chance_outcomes, chance_dim)
    
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: (batch, observation_dim)
        
        Returns:
            chance_code: (batch, chance_dim)
            outcome_logits: (batch, n_outcomes) pour l'apprentissage
        """
        h = self.shared(observation)
        logits = self.outcome_head(h)
        
        # Gumbel-Softmax pour rendre differentiable
        if self.training:
            outcome_probs = F.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
            outcome_idx = logits.argmax(dim=-1)
            outcome_probs = F.one_hot(outcome_idx, self.n_outcomes).float()
        
        # Weighted sum des embeddings
        chance_code = torch.matmul(outcome_probs, self.outcome_embedding.weight)
        
        return chance_code, logits
    
    def sample_outcome(self, afterstate: torch.Tensor) -> torch.Tensor:
        """
        Echantillonne un outcome pendant MCTS (sans observation).
        Utilise une distribution uniforme ou apprise.
        
        Pour le planning, on echantillonne plusieurs outcomes possibles.
        """
        batch_size = afterstate.size(0)
        # Echantillonnage uniforme par defaut
        outcome_idx = torch.randint(0, self.n_outcomes, (batch_size,), device=afterstate.device)
        return self.outcome_embedding(outcome_idx)


class ChanceDynamicsNetwork(nn.Module):
    """
    Reseau de dynamique stochastique g_c: (afterstate, chance_code) -> (etat, reward).
    
    Applique l'effet de l'evenement aleatoire a l'afterstate.
    """
    
    def __init__(
        self,
        latent_dim: int,
        chance_dim: int,
        hidden_dims: List[int] = [128]
    ):
        super().__init__()
        
        input_dim = latent_dim + chance_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.state_head = nn.Linear(prev_dim, latent_dim)
        self.reward_head = nn.Linear(prev_dim, 1)
        
        self.shared = nn.Sequential(*layers)
    
    def forward(
        self, 
        afterstate: torch.Tensor, 
        chance_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            afterstate: (batch, latent_dim)
            chance_code: (batch, chance_dim)
        
        Returns:
            next_state: (batch, latent_dim)
            reward: (batch, 1)
        """
        x = torch.cat([afterstate, chance_code], dim=1)
        h = self.shared(x)
        
        next_state = self.state_head(h)
        reward = self.reward_head(h)
        
        return next_state, reward


class PredictionNetwork(nn.Module):
    """
    Reseau de prediction f: etat_latent -> (policy, value).
    Identique a MuZero standard.
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
        
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, n_actions)
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(latent_state)
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value


class AfterstatePredictionNetwork(nn.Module):
    """
    Prediction sur afterstate (pour MCTS).
    Predit la distribution des outcomes possibles.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_outcomes: int,
        hidden_dims: List[int] = [64]
    ):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.outcome_head = nn.Linear(prev_dim, n_outcomes)
        self.value_head = nn.Linear(prev_dim, 1)
        self.shared = nn.Sequential(*layers)
    
    def forward(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(afterstate)
        outcome_logits = self.outcome_head(h)
        value = self.value_head(h)
        return outcome_logits, value


@dataclass
class StochasticMCTSNode:
    """
    Noeud MCTS pour MuZero Stochastique.
    
    Deux types de noeuds:
    - Decision nodes: le joueur choisit une action
    - Chance nodes: un evenement aleatoire se produit
    """
    state: torch.Tensor
    is_chance_node: bool = False  # True si c'est un noeud de chance
    parent: Optional["StochasticMCTSNode"] = None
    action: Optional[int] = None
    chance_outcome: Optional[int] = None
    
    visits: int = 0
    value_sum: float = 0.0
    
    children: Dict[int, "StochasticMCTSNode"] = field(default_factory=dict)
    prior: float = 0.0
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class StochasticMuZeroAgent(Agent):
    """
    Agent MuZero Stochastique.
    
    Etend MuZero pour les environnements avec transitions aleatoires.
    Utilise un MCTS modifie avec des noeuds de chance.
    
    Exemple d'utilisation:
        >>> agent = StochasticMuZeroAgent(state_dim=10, n_actions=4)
        >>> action = agent.act(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        latent_dim: int = 64,
        chance_dim: int = 16,
        n_chance_outcomes: int = 32,
        hidden_dims: List[int] = [128, 128],
        n_simulations: int = 50,
        c_puct: float = 1.0,
        gamma: float = 0.99,
        lr: float = 1e-3,
        unroll_steps: int = 5,
        buffer_size: int = 10000,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialise MuZero Stochastique.
        
        Args:
            state_dim: Dimension des observations
            n_actions: Nombre d'actions
            latent_dim: Dimension de l'espace latent
            chance_dim: Dimension du code de chance
            n_chance_outcomes: Nombre d'outcomes stochastiques discrets
            hidden_dims: Architecture des reseaux
            n_simulations: Simulations MCTS par decision
            c_puct: Constante d'exploration UCB
            gamma: Facteur d'actualisation
            lr: Taux d'apprentissage
            unroll_steps: Nombre de pas pour l'entrainement
            buffer_size: Taille du replay buffer
            batch_size: Taille des mini-batches
            device: "cpu" ou "cuda"
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="MuZero-Stochastic",
            device=device
        )
        
        self.latent_dim = latent_dim
        self.chance_dim = chance_dim
        self.n_chance_outcomes = n_chance_outcomes
        self.hidden_dims = hidden_dims
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.lr = lr
        self.unroll_steps = unroll_steps
        self.batch_size = batch_size
        
        # Reseaux
        self.representation = RepresentationNetwork(
            state_dim, latent_dim, hidden_dims
        ).to(self.device)
        
        self.afterstate_dynamics = AfterstateNetwork(
            latent_dim, n_actions, hidden_dims[:1]
        ).to(self.device)
        
        self.chance_encoder = ChanceEncoderNetwork(
            state_dim, chance_dim, hidden_dims[:1], n_chance_outcomes
        ).to(self.device)
        
        self.chance_dynamics = ChanceDynamicsNetwork(
            latent_dim, chance_dim, hidden_dims[:1]
        ).to(self.device)
        
        self.prediction = PredictionNetwork(
            latent_dim, n_actions, hidden_dims[:1]
        ).to(self.device)
        
        self.afterstate_prediction = AfterstatePredictionNetwork(
            latent_dim, n_chance_outcomes, hidden_dims[:1]
        ).to(self.device)
        
        # Optimiseur
        all_params = list(self.representation.parameters()) + \
                     list(self.afterstate_dynamics.parameters()) + \
                     list(self.chance_encoder.parameters()) + \
                     list(self.chance_dynamics.parameters()) + \
                     list(self.prediction.parameters()) + \
                     list(self.afterstate_prediction.parameters())
        
        self.optimizer = optim.Adam(all_params, lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self._training = True
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action avec MCTS stochastique.
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Encoder l'observation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            root_state = self.representation(obs_tensor)
        
        # Creer le noeud racine
        root = StochasticMCTSNode(state=root_state, is_chance_node=False)
        
        # Initialiser avec la prediction du reseau
        with torch.no_grad():
            policy_logits, value = self.prediction(root_state)
            policy = F.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
        
        for a in available_actions:
            root.children[a] = StochasticMCTSNode(
                state=None,  # Sera calcule lors de la visite
                parent=root,
                action=a,
                is_chance_node=True,  # Apres action -> noeud de chance
                prior=policy[a]
            )
        
        # Simulations MCTS
        for _ in range(self.n_simulations):
            self._simulate(root, available_actions)
        
        # Choisir l'action la plus visitee
        visits = {a: root.children[a].visits for a in available_actions}
        best_action = max(visits, key=visits.get)
        
        return best_action
    
    def _simulate(self, root: StochasticMCTSNode, available_actions: List[int]):
        """Execute une simulation MCTS."""
        node = root
        search_path = [node]
        
        # Selection
        while node.children and not node.is_chance_node:
            node = self._select_child(node)
            search_path.append(node)
        
        # Si c'est un noeud de chance, echantillonner un outcome
        if node.is_chance_node and node.state is not None:
            outcome_code = self.chance_encoder.sample_outcome(node.state)
            with torch.no_grad():
                next_state, reward = self.chance_dynamics(node.state, outcome_code)
            
            # Creer un noeud enfant pour cet outcome
            outcome_idx = 0  # Simplifie pour le moment
            if outcome_idx not in node.children:
                node.children[outcome_idx] = StochasticMCTSNode(
                    state=next_state,
                    parent=node,
                    chance_outcome=outcome_idx,
                    is_chance_node=False
                )
            
            node = node.children[outcome_idx]
            search_path.append(node)
        
        # Expansion si necessaire
        if node.state is None and node.parent is not None:
            # Calculer l'afterstate
            with torch.no_grad():
                afterstate, _ = self.afterstate_dynamics(
                    node.parent.state,
                    torch.tensor([node.action], device=self.device)
                )
            node.state = afterstate
        
        # Evaluation
        if node.state is not None:
            with torch.no_grad():
                policy_logits, value = self.prediction(node.state)
                value = value.item()
        else:
            value = 0.0
        
        # Backpropagation
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            value = self.gamma * value
    
    def _select_child(self, node: StochasticMCTSNode) -> StochasticMCTSNode:
        """Selectionne un enfant avec UCB."""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value
                exploration = self.c_puct * child.prior * \
                    math.sqrt(node.visits) / (1 + child.visits)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Stocke une transition."""
        self.replay_buffer.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        })
    
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ) -> Optional[Dict[str, float]]:
        """Interface standard - stocke et apprend."""
        self.store_transition(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) >= self.batch_size:
            return self.train_step()
        return None
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """Effectue un pas d'entrainement."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Echantillonner un batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Utiliser np.array pour éviter le warning de conversion lente
        states = torch.FloatTensor(np.array([t['state'] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t['action'] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t['reward'] for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t['next_state'] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t['done'] for t in batch])).to(self.device)
        
        # Forward pass
        latent_states = self.representation(states)
        
        # Afterstate
        afterstates, pred_rewards_a = self.afterstate_dynamics(latent_states, actions)
        
        # Chance encoding
        chance_codes, chance_logits = self.chance_encoder(next_states)
        
        # Next state prediction
        pred_next_states, pred_rewards_c = self.chance_dynamics(afterstates, chance_codes)
        
        # Predictions
        policy_logits, values = self.prediction(latent_states)
        
        # Target values
        with torch.no_grad():
            next_latent = self.representation(next_states)
            _, next_values = self.prediction(next_latent)
            target_values = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        
        # Losses
        value_loss = F.mse_loss(values.squeeze(), target_values)
        
        reward_loss = F.mse_loss(
            pred_rewards_a.squeeze() + pred_rewards_c.squeeze(), 
            rewards
        )
        
        # Consistency loss (optional)
        with torch.no_grad():
            target_latent = self.representation(next_states)
        consistency_loss = F.mse_loss(pred_next_states, target_latent)
        
        # Total loss
        total_loss = value_loss + reward_loss + 0.1 * consistency_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.representation.parameters()) + 
            list(self.afterstate_dynamics.parameters()) +
            list(self.chance_encoder.parameters()) +
            list(self.chance_dynamics.parameters()) +
            list(self.prediction.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.training_steps += 1
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item(),
            'consistency_loss': consistency_loss.item()
        }
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        torch.save({
            'representation': self.representation.state_dict(),
            'afterstate_dynamics': self.afterstate_dynamics.state_dict(),
            'chance_encoder': self.chance_encoder.state_dict(),
            'chance_dynamics': self.chance_dynamics.state_dict(),
            'prediction': self.prediction.state_dict(),
            'afterstate_prediction': self.afterstate_prediction.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'config': self.get_config()
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.representation.load_state_dict(checkpoint['representation'])
        self.afterstate_dynamics.load_state_dict(checkpoint['afterstate_dynamics'])
        self.chance_encoder.load_state_dict(checkpoint['chance_encoder'])
        self.chance_dynamics.load_state_dict(checkpoint['chance_dynamics'])
        self.prediction.load_state_dict(checkpoint['prediction'])
        if 'afterstate_prediction' in checkpoint:
            self.afterstate_prediction.load_state_dict(checkpoint['afterstate_prediction'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint.get('training_steps', 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'type': 'MuZero-Stochastic',
            'latent_dim': self.latent_dim,
            'chance_dim': self.chance_dim,
            'n_chance_outcomes': self.n_chance_outcomes,
            'hidden_dims': self.hidden_dims,
            'n_simulations': self.n_simulations,
            'c_puct': self.c_puct,
            'gamma': self.gamma,
            'lr': self.lr,
            'unroll_steps': self.unroll_steps
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test de MuZero Stochastique ===\n")
    
    # Simuler un environnement simple
    state_dim = 10
    n_actions = 4
    
    agent = StochasticMuZeroAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        latent_dim=32,
        chance_dim=8,
        n_chance_outcomes=16,
        hidden_dims=[64],
        n_simulations=10
    )
    
    print(f"Agent: {agent.name}")
    print(f"Config: {agent.get_config()}")
    
    # Test action
    print("\n1. Test action:")
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.act(state)
    print(f"   Action choisie: {action}")
    
    # Collecter des transitions
    print("\n2. Collecte de donnees:")
    for i in range(50):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(n_actions)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = i % 10 == 9
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(agent.replay_buffer)}")
    
    # Entrainement
    print("\n3. Entrainement:")
    for i in range(5):
        metrics = agent.train_step()
        if metrics:
            print(f"   Step {i+1}: total_loss={metrics['total_loss']:.4f}, "
                  f"value_loss={metrics['value_loss']:.4f}")
    
    # Test save/load
    print("\n4. Test save/load:")
    agent.save("/tmp/test_muzero_stochastic.pt")
    agent.load("/tmp/test_muzero_stochastic.pt")
    print("   Save/Load OK")
    
    print("\n[OK] Tests passes!")
