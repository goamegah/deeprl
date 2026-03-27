"""
AlphaZero - MCTS guidé par réseau de neurones.

AlphaZero (Silver et al., 2017) combine:
- MCTS pour la recherche (exploration structurée)
- Un réseau de neurones pour policy et value (apprentissage)

Différences avec MCTS standard:
- La policy du réseau guide l'exploration (prior P(s,a))
- La value du réseau remplace les rollouts aléatoires
- L'arbre est construit de manière plus intelligente

Formule PUCT (Polynomial UCT):
    a = argmax_a [ Q(s,a) + c_puct * P(s,a) * √N(s) / (1 + N(s,a)) ]

Entraînement:
1. Self-play: jouer des parties contre soi-même
2. Collecter (state, policy_cible, value_cible)
3. Entraîner le réseau sur ces données
4. Répéter

Références:
- "Mastering the Game of Go without Human Knowledge" (Silver et al., 2017)
- "Mastering Chess and Shogi by Self-Play" (Silver et al., 2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import math

from deeprl.agents.base import Agent
from deeprl.envs.base import Environment


class AlphaZeroNetwork(nn.Module):
    """
    Réseau de neurones pour AlphaZero.
    
    Architecture:
    - Tronc commun (MLP ou CNN)
    - Tête Policy: softmax sur les actions
    - Tête Value: scalaire [-1, 1]
    
    Sorties:
    - policy: distribution sur les actions
    - value: estimation de la valeur de l'état
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [256, 256],
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Tronc commun
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Tête Policy
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # Tête Value
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Valeur dans [-1, 1]
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: États (batch_size, state_dim)
        
        Returns:
            policy_logits: (batch_size, n_actions)
            value: (batch_size, 1)
        """
        trunk_out = self.trunk(x)
        
        policy_logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)
        
        return policy_logits, value
    
    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Prédit policy et value pour un seul état.
        
        Args:
            x: État (state_dim,) ou (1, state_dim)
            mask: Masque des actions valides
        
        Returns:
            (policy_probs, value)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Passer en mode évaluation pour BatchNorm
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            logits, value = self.forward(x)
            
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))
            
            policy = F.softmax(logits, dim=-1)
        
        # Restaurer le mode
        if was_training:
            self.train()
        
        return policy.cpu().numpy()[0], value.item()


@dataclass
class AlphaZeroNode:
    """Nœud de l'arbre MCTS pour AlphaZero."""
    
    state: np.ndarray
    parent: Optional["AlphaZeroNode"] = None
    action: Optional[int] = None
    player: int = 0
    
    # Statistiques
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # P(s, a) du réseau
    
    # Enfants
    children: Dict[int, "AlphaZeroNode"] = field(default_factory=dict)
    is_expanded: bool = False
    is_terminal: bool = False
    
    @property
    def value(self) -> float:
        """Valeur moyenne Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int, parent_player: int = 0) -> float:
        """
        Score PUCT pour la sélection.
        
        Chaque noeud stocke la valeur du point de vue de SON propre joueur.
        Le parent veut maximiser depuis SA perspective, donc on inverse
        quand les joueurs diffèrent.
        
        PUCT = Q(s,a) + c_puct * P(s,a) * √N(parent) / (1 + N(s,a))
        """
        exploitation = self.value
        if self.player != parent_player:
            exploitation = -exploitation
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration
    
    def select_child(self, c_puct: float) -> "AlphaZeroNode":
        """Sélectionne l'enfant avec le meilleur score PUCT."""
        return max(
            self.children.values(),
            key=lambda n: n.ucb_score(c_puct, self.visit_count, self.player)
        )
    
    def best_action(self, temperature: float = 0.0) -> int:
        """
        Retourne l'action selon les visites.
        
        temperature=0: argmax (exploitation)
        temperature>0: échantillonnage proportionnel aux visites
        """
        visits = np.array([
            self.children[a].visit_count if a in self.children else 0
            for a in range(max(self.children.keys()) + 1)
        ])
        
        if temperature == 0:
            return int(np.argmax(visits))
        else:
            # Échantillonnage avec température
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            return int(np.random.choice(len(probs), p=probs))
    
    def get_policy(self, n_actions: int) -> np.ndarray:
        """Retourne la politique (distribution des visites)."""
        policy = np.zeros(n_actions, dtype=np.float32)
        total = sum(c.visit_count for c in self.children.values())
        
        if total > 0:
            for action, child in self.children.items():
                policy[action] = child.visit_count / total
        
        return policy


class AlphaZeroAgent(Agent):
    """
    Agent AlphaZero.
    
    Combine MCTS avec un réseau de neurones (policy + value).
    
    Utilisation:
    1. Pour jouer: utilise MCTS guidé par le réseau
    2. Pour s'entraîner: self-play puis apprentissage supervisé
    
    Exemple:
        >>> agent = AlphaZeroAgent(state_dim=114, n_actions=16)
        >>> action = agent.act(state, env=env)
        >>> # Self-play et entraînement
        >>> examples = agent.self_play(env, n_games=100)
        >>> agent.train(examples)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        # Architecture réseau
        hidden_dims: List[int] = [256, 256],
        # MCTS
        n_simulations: int = 100,
        c_puct: float = 1.0,
        # Entraînement
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Self-play
        temperature: float = 1.0,
        temperature_threshold: int = 15,  # Après N coups, temp=0
        # Autres
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent AlphaZero.
        
        Args:
            state_dim: Dimension de l'état
            n_actions: Nombre d'actions
            hidden_dims: Architecture du réseau
            n_simulations: Nombre de simulations MCTS par coup
            c_puct: Constante d'exploration PUCT
            lr: Learning rate
            weight_decay: Régularisation L2
            temperature: Température pour l'exploration (self-play)
            temperature_threshold: Après N coups, température = 0
            device: "cpu" ou "cuda"
            seed: Graine aléatoire
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="AlphaZero",
            device=device
        )
        
        self.hidden_dims = hidden_dims
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Réseau de neurones
        self.network = AlphaZeroNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Compteur de coups pour la température
        self._move_count = 0
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = False,
        env: Optional[Environment] = None,
        **kwargs
    ) -> int:
        """
        Choisit une action via MCTS guidé par le réseau.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: Mode entraînement (utilise température)
            env: Environnement (requis pour MCTS)
        
        Returns:
            Action choisie
        """
        if env is None:
            raise ValueError("AlphaZero nécessite un environnement (env=...)")
        
        if available_actions is None:
            available_actions = env.get_available_actions()
        
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Créer la racine
        root = AlphaZeroNode(
            state=state.copy(),
            player=env.current_player
        )
        
        # Expand la racine
        self._expand(root, env, available_actions)
        
        # Simulations MCTS
        for _ in range(self.n_simulations):
            node = root
            env_sim = env.clone()
            path = [node]
            
            # Selection
            while node.is_expanded and not node.is_terminal:
                node = node.select_child(self.c_puct)
                path.append(node)
                
                if node.action is not None:
                    _, _, done = env_sim.step(node.action)
                    if done:
                        node.is_terminal = True
                        break
            
            # Expansion
            if not node.is_terminal and not node.is_expanded:
                available = env_sim.get_available_actions()
                if len(available) > 0:
                    self._expand(node, env_sim, available)
            
            # Évaluation (value du point de vue du noeud courant)
            if node.is_terminal:
                # Utiliser la récompense réelle
                if hasattr(env_sim, '_winner') and env_sim._winner is not None:
                    if env_sim._winner == node.player:
                        value = 1.0
                    else:
                        value = -1.0
                else:
                    value = 0.0  # Nul
            else:
                state_tensor = torch.FloatTensor(node.state).to(self.device)
                _, value = self.network.predict(state_tensor)
                # Le réseau prédit depuis la perspective du joueur courant
                # → pas d'ajustement nécessaire
            
            # Backpropagation (valeur du point de vue de chaque noeud)
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                # Inverser seulement quand le joueur change entre parent et enfant
                if n.parent is not None and n.player != n.parent.player:
                    value = -value
        
        # Choisir l'action
        temp = self.temperature if training and self._move_count < self.temperature_threshold else 0.0
        action = root.best_action(temperature=temp)
        
        self._move_count += 1
        
        return action
    
    def _expand(
        self,
        node: AlphaZeroNode,
        env: Environment,
        available_actions: List[int]
    ):
        """
        Expand un nœud en ajoutant ses enfants.
        
        Utilise le réseau pour obtenir les priors P(s, a).
        """
        state_tensor = torch.FloatTensor(node.state).to(self.device)
        
        # Créer le masque des actions valides
        mask = torch.zeros(self.n_actions, dtype=torch.bool).to(self.device)
        for a in available_actions:
            mask[a] = True
        
        policy, value = self.network.predict(state_tensor, mask)
        
        # Créer les enfants
        for action in available_actions:
            child_env = env.clone()
            next_state, reward, done = child_env.step(action)
            
            child = AlphaZeroNode(
                state=next_state.copy(),
                parent=node,
                action=action,
                player=child_env.current_player,
                prior=policy[action],
                is_terminal=done
            )
            node.children[action] = child
        
        node.is_expanded = True
    
    def get_action_probs(
        self,
        state: np.ndarray,
        env: Environment,
        available_actions: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Retourne la politique MCTS et la value.
        
        Returns:
            (policy, value)
        """
        if available_actions is None:
            available_actions = env.get_available_actions()
        
        # Créer la racine
        root = AlphaZeroNode(
            state=state.copy(),
            player=env.current_player
        )
        
        # Expand
        self._expand(root, env, available_actions)
        
        # Simulations
        for _ in range(self.n_simulations):
            node = root
            env_sim = env.clone()
            path = [node]
            
            while node.is_expanded and not node.is_terminal:
                node = node.select_child(self.c_puct)
                path.append(node)
                
                if node.action is not None:
                    _, _, done = env_sim.step(node.action)
                    if done:
                        node.is_terminal = True
                        break
            
            if not node.is_terminal and not node.is_expanded:
                available = env_sim.get_available_actions()
                if len(available) > 0:
                    self._expand(node, env_sim, available)
            
            if node.is_terminal:
                if hasattr(env_sim, '_winner') and env_sim._winner is not None:
                    if env_sim._winner == node.player:
                        value = 1.0
                    else:
                        value = -1.0
                else:
                    value = 0.0  # Nul ou pas de winner
            else:
                state_tensor = torch.FloatTensor(node.state).to(self.device)
                _, value = self.network.predict(state_tensor)
            
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                if n.parent is not None and n.player != n.parent.player:
                    value = -value
        
        policy = root.get_policy(self.n_actions)
        root_value = root.value
        
        return policy, root_value
    
    def self_play(
        self,
        env: Environment,
        n_games: int = 100,
        verbose: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Génère des données par self-play.
        
        Args:
            env: Environnement de jeu
            n_games: Nombre de parties
            verbose: Afficher la progression
        
        Returns:
            Liste de (state, policy, value) pour l'entraînement
        """
        examples = []
        wins = [0, 0, 0]  # J1, J2, Nul
        
        for game in range(n_games):
            game_examples = []
            state = env.reset()
            self._move_count = 0
            
            while not env.is_game_over:
                available = env.get_available_actions()
                
                # Obtenir la politique MCTS
                policy, _ = self.get_action_probs(state, env, available)
                
                # Stocker l'exemple (la value sera ajoutée à la fin)
                game_examples.append((
                    state.copy(),
                    policy.copy(),
                    env.current_player
                ))
                
                # Choisir l'action
                temp = self.temperature if self._move_count < self.temperature_threshold else 0.0
                
                if temp > 0:
                    action_probs = policy ** (1.0 / temp)
                    action_probs /= action_probs.sum()
                    action = np.random.choice(len(action_probs), p=action_probs)
                else:
                    action = int(np.argmax(policy))
                
                state, reward, done = env.step(action)
                self._move_count += 1
            
            # Déterminer le résultat
            if hasattr(env, '_winner') and env._winner is not None:
                winner = env._winner
            else:
                winner = -1  # Nul ou pas de winner
            
            if winner == 0:
                wins[0] += 1
            elif winner == 1:
                wins[1] += 1
            else:
                wins[2] += 1
            
            # Ajouter les exemples avec la valeur finale
            for state, policy, player in game_examples:
                if winner == player:
                    value = 1.0
                elif winner >= 0:
                    value = -1.0
                else:
                    value = 0.0
                
                examples.append((state, policy, value))
            
            if verbose and (game + 1) % 10 == 0:
                print(f"Self-play: {game + 1}/{n_games} parties")
                print(f"  J1: {wins[0]}, J2: {wins[1]}, Nuls: {wins[2]}")
        
        return examples
    
    def train_on_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray, float]],
        batch_size: int = 32,
        n_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Entraîne le réseau sur les exemples de self-play.
        
        Args:
            examples: Liste de (state, policy, value)
            batch_size: Taille des batches
            n_epochs: Nombre d'epochs
        
        Returns:
            Métriques d'entraînement
        """
        self.network.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(examples)
            
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]
                
                # Utiliser np.array pour éviter le warning de conversion lente
                states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
                target_policies = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
                target_values = torch.FloatTensor(np.array([[e[2]] for e in batch])).to(self.device)
                
                # Forward
                policy_logits, values = self.network(states)
                
                # Policy loss (cross-entropy)
                log_probs = F.log_softmax(policy_logits, dim=-1)
                policy_loss = -torch.sum(target_policies * log_probs) / len(batch)
                
                # Value loss (MSE)
                value_loss = F.mse_loss(values, target_values)
                
                # Total loss
                loss = policy_loss + value_loss
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_updates += 1
        
        self.training_steps += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
        }
    
    def learn(self, *args, **kwargs) -> None:
        """Interface standard (non utilisée pour AlphaZero)."""
        return None
    
    def on_episode_start(self):
        """Reset le compteur de coups."""
        self._move_count = 0
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "config": self.get_config(),
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint.get("training_steps", 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            "type": "AlphaZero",
            "hidden_dims": self.hidden_dims,
            "n_simulations": self.n_simulations,
            "c_puct": self.c_puct,
            "lr": self.lr,
            "temperature": self.temperature,
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test d'AlphaZero ===\n")
    
    from deeprl.envs.tictactoe import TicTacToe
    
    env = TicTacToe()
    agent = AlphaZeroAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[64, 64],
        n_simulations=25
    )
    
    print(f"Agent: {agent.name}")
    print(f"Config: {agent.get_config()}")
    
    # Tester l'action
    print("\n1. Test action:")
    state = env.reset()
    action = agent.act(state, env=env)
    print(f"   Action choisie: {action}")
    
    # Tester get_action_probs
    print("\n2. Test policy MCTS:")
    policy, value = agent.get_action_probs(state, env)
    print(f"   Policy: {policy}")
    print(f"   Value: {value:.3f}")
    
    # Tester une partie
    print("\n3. Partie de test:")
    state = env.reset()
    while not env.is_game_over:
        available = env.get_available_actions()
        action = agent.act(state, available, env=env)
        state, _, done = env.step(action)
    
    env.render()
    
    # Self-play rapide
    print("\n4. Self-play (3 parties):")
    examples = agent.self_play(env, n_games=3, verbose=True)
    print(f"   Exemples générés: {len(examples)}")
    
    # Entraînement
    print("\n5. Entraînement:")
    metrics = agent.train_on_examples(examples, batch_size=8, n_epochs=2)
    print(f"   Policy loss: {metrics['policy_loss']:.4f}")
    print(f"   Value loss: {metrics['value_loss']:.4f}")
    
    print("\n[OK] Tests passes!")
