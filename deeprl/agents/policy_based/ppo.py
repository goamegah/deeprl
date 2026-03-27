"""
PPO - Proximal Policy Optimization.

PPO (Schulman et al., 2017) est l'un des algorithmes de RL les plus
utilisés aujourd'hui. Il combine les avantages de:
- Policy Gradient (apprentissage de politique)
- Actor-Critic (réduction de variance)
- Trust Region (stabilité)

Idée clé:
- Limiter la taille des mises à jour de politique
- Éviter les changements trop brutaux qui détruisent l'apprentissage

Deux variantes:
1. PPO-Penalty: ajoute une pénalité KL à la loss
2. PPO-Clip: clippe le ratio de probabilité (plus simple, plus utilisé)

Équation (PPO-Clip):
    L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

où r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) est le ratio de probabilité.

Référence:
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Optional, Dict, Any, Tuple

from deeprl.agents.base import Agent
from deeprl.networks.mlp import ActorCriticMLP


class RolloutBuffer:
    """
    Buffer pour stocker les trajectoires collectées.
    
    Stocke les données nécessaires pour PPO:
    - states, actions, rewards
    - log_probs (pour calculer le ratio)
    - values (pour calculer les avantages)
    - action_masks (pour masquer les actions invalides lors de l'update)
    - advantages, returns (calculés après collecte)
    """
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.action_masks: List[Optional[np.ndarray]] = []
        
        # Calculés après la collecte
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        action_mask: Optional[np.ndarray] = None
    ):
        """Ajoute une transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Calcule les returns et avantages avec GAE.
        
        GAE (Generalized Advantage Estimation) réduit la variance
        tout en contrôlant le biais.
        
        Args:
            last_value: V(s_T+1) pour le bootstrap
            gamma: Facteur d'actualisation
            gae_lambda: Paramètre GAE (0=TD, 1=MC)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        # où δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        last_gae = 0
        last_val = last_value
        
        for t in reversed(range(n)):
            if self.dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = last_val
            
            # TD error
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            
            # GAE
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
            
            # Return = advantage + value
            returns[t] = advantages[t] + self.values[t]
            
            last_val = self.values[t]
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(
        self,
        batch_size: int,
        device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Génère des mini-batches pour l'entraînement.
        
        Args:
            batch_size: Taille des batches
            device: Device PyTorch
        
        Yields:
            Dictionnaires avec les tenseurs
        """
        n = len(self.states)
        indices = np.random.permutation(n)
        
        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                "states": torch.FloatTensor(
                    np.array([self.states[i] for i in batch_indices])
                ).to(device),
                "actions": torch.LongTensor(
                    [self.actions[i] for i in batch_indices]
                ).to(device),
                "old_log_probs": torch.FloatTensor(
                    [self.log_probs[i] for i in batch_indices]
                ).to(device),
                "advantages": torch.FloatTensor(
                    self.advantages[batch_indices]
                ).to(device),
                "returns": torch.FloatTensor(
                    self.returns[batch_indices]
                ).to(device),
            }
            
            # Action masks: shape (batch_size, n_actions), True = valid
            # If ANY transition has a mask, build the tensor; else None
            batch_masks = [self.action_masks[i] for i in batch_indices]
            if any(m is not None for m in batch_masks):
                mask_array = np.ones((len(batch_indices), self.n_actions), dtype=np.float32)
                for j, m in enumerate(batch_masks):
                    if m is not None:
                        mask_array[j] = m
                batch["action_masks"] = torch.FloatTensor(mask_array).to(device)
            else:
                batch["action_masks"] = None
            
            batches.append(batch)
        
        return batches
    
    def clear(self):
        """Vide le buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()
        self.advantages = None
        self.returns = None
    
    def __len__(self):
        return len(self.states)


class PPOAgent(Agent):
    """
    Agent PPO (Proximal Policy Optimization).
    
    Utilise un réseau Actor-Critic et la méthode de clipping
    pour des mises à jour stables.
    
    Caractéristiques:
    - Actor-Critic avec couches partagées ou séparées
    - GAE pour l'estimation des avantages
    - Clipping du ratio de probabilité
    - Plusieurs epochs sur les mêmes données
    
    Exemple d'utilisation:
        >>> agent = PPOAgent(state_dim=4, n_actions=2)
        >>> # Collecter des trajectoires
        >>> for _ in range(n_steps):
        >>>     action, log_prob, value = agent.act_with_value(state)
        >>>     agent.store(state, action, reward, log_prob, value, done)
        >>> # Mettre à jour
        >>> agent.update()
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        # Architecture
        hidden_dims: List[int] = [64, 64],
        shared_layers: bool = True,
        # Hyperparamètres PPO
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        # Entraînement
        n_epochs: int = 4,
        batch_size: int = 64,
        # Coefficients de loss
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        # Autres
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent PPO.
        
        Args:
            state_dim: Dimension de l'état
            n_actions: Nombre d'actions
            hidden_dims: Architecture des couches cachées
            shared_layers: Partager les couches entre Actor et Critic
            lr: Learning rate
            gamma: Facteur d'actualisation
            gae_lambda: Paramètre GAE
            clip_epsilon: Paramètre de clipping (typiquement 0.1-0.3)
            n_epochs: Nombre de passes sur les données
            batch_size: Taille des mini-batches
            value_coef: Coefficient de la loss du Critic
            entropy_coef: Coefficient du bonus d'entropie
            max_grad_norm: Clipping des gradients
            device: "cpu" ou "cuda"
            seed: Graine aléatoire
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="PPO",
            device=device
        )
        
        # Sauvegarder les hyperparamètres
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Réseau Actor-Critic
        self.network = ActorCriticMLP(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            shared=shared_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Buffer pour les trajectoires
        self.buffer = RolloutBuffer(n_actions=n_actions)
        
        self._training = True
        
        # Temporary storage from act() for use in learn()
        self._last_log_prob: Optional[float] = None
        self._last_value: Optional[float] = None
        self._last_action_mask: Optional[np.ndarray] = None
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: Mode entraînement
        
        Returns:
            Action choisie
        """
        action, log_prob, value = self.act_with_value(state, available_actions, training)
        
        # Save for learn() to reuse (avoids recomputation AND ensures
        # the log_prob/value are consistent with the mask used here)
        self._last_log_prob = log_prob
        self._last_value = value
        if available_actions is not None:
            mask = np.zeros(self.n_actions, dtype=np.float32)
            for a in available_actions:
                mask[a] = 1.0
            self._last_action_mask = mask
        else:
            self._last_action_mask = None
        
        return action
    
    def act_with_value(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Choisit une action et retourne aussi log_prob et value.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: Mode entraînement
        
        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.network(state_tensor)
            
            # Remplacer NaN par 0 (distribution uniforme) si le réseau diverge
            if torch.isnan(policy_logits).any():
                policy_logits = torch.zeros_like(policy_logits)
            
            # Masquer les actions non disponibles
            if available_actions is not None:
                mask = torch.full((self.n_actions,), float('-inf')).to(self.device)
                for a in available_actions:
                    mask[a] = 0
                policy_logits = policy_logits + mask
            
            # Stabilité numérique
            policy_logits = policy_logits.clamp(-50, 50)
            probs = torch.softmax(policy_logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            dist = Categorical(probs)
            
            use_training = training and self._training
            if use_training:
                action = dist.sample()
            else:
                action = probs.argmax(dim=-1)
            
            log_prob = dist.log_prob(action)
            
            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item())
            )
    
    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        action_mask: Optional[np.ndarray] = None
    ):
        """Stocke une transition dans le buffer."""
        self.buffer.push(state, action, reward, log_prob, value, done, action_mask)
    
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
        Interface standard.
        
        PPO collecte d'abord des trajectoires puis met à jour.
        Cette méthode stocke et met à jour si done=True.
        
        Uses log_prob, value, and action_mask saved by act() to ensure
        consistency with the masked distribution used during action selection.
        """
        # Use values saved by act() if available (normal flow: act → learn)
        if self._last_log_prob is not None:
            log_prob = self._last_log_prob
            value = self._last_value
            action_mask = self._last_action_mask
            # Clear after use
            self._last_log_prob = None
            self._last_value = None
            self._last_action_mask = None
        else:
            # Fallback: recompute (e.g., if learn() called without prior act())
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                policy_logits, val = self.network(state_tensor)
                
                if torch.isnan(policy_logits).any():
                    policy_logits = torch.zeros_like(policy_logits)
                if torch.isnan(val).any():
                    val = torch.zeros_like(val)
                
                # Try to get available_actions from kwargs
                available_actions = kwargs.get('available_actions')
                if available_actions is not None:
                    mask_t = torch.full((self.n_actions,), float('-inf')).to(self.device)
                    for a in available_actions:
                        mask_t[a] = 0
                    policy_logits = policy_logits + mask_t
                    action_mask = np.zeros(self.n_actions, dtype=np.float32)
                    for a in available_actions:
                        action_mask[a] = 1.0
                else:
                    action_mask = None
                
                policy_logits = policy_logits.clamp(-50, 50)
                dist = Categorical(logits=policy_logits)
                lp = dist.log_prob(torch.tensor(action).to(self.device))
                log_prob = float(lp.item())
                value = float(val.item())
        
        self.store(state, action, reward, log_prob, value, done, action_mask)
        
        # Mettre à jour quand l'épisode est terminé OU quand le buffer
        # atteint une taille suffisante (rollout fixe, comme dans le
        # papier PPO — ne pas attendre done pour les longs épisodes)
        if done:
            return self.update(last_value=0.0)
        elif len(self.buffer) >= self.batch_size * self.n_epochs:
            # Bootstrap V(s') pour l'état courant (pas terminé)
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.network(next_tensor)
            return self.update(last_value=float(next_value.item()))
        return None
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Effectue la mise à jour PPO.
        
        Args:
            last_value: V(s_T+1) pour le bootstrap (0 si terminé)
        
        Returns:
            Métriques d'entraînement
        """
        if len(self.buffer) == 0:
            return {}
        
        # Calculer returns et avantages
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Normaliser les avantages
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
        # Métriques agrégées
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # Plusieurs epochs sur les mêmes données
        for epoch in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size, self.device)
            
            for batch in batches:
                loss_info = self._update_batch(batch)
                total_policy_loss += loss_info["policy_loss"]
                total_value_loss += loss_info["value_loss"]
                total_entropy += loss_info["entropy"]
                n_updates += 1
        
        # Nettoyer le buffer
        self.buffer.clear()
        
        self.training_steps += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def _update_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Mise à jour sur un mini-batch.
        
        Args:
            batch: Dictionnaire avec les tenseurs
        
        Returns:
            Métriques du batch
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        action_masks = batch["action_masks"]  # (batch, n_actions) float or None
        
        # Forward pass
        policy_logits, values = self.network(states)
        values = values.squeeze(-1)
        
        # Détection NaN dans le forward pass — skip le batch si corrompu
        if torch.isnan(policy_logits).any() or torch.isnan(values).any():
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Apply action masks: set unavailable actions to -inf
        # CRITICAL for envs with variable action spaces (Quarto, etc.)
        # Without this, ratio = exp(new_log_prob - old_log_prob) is wrong
        # because old_log_probs were computed under a masked distribution.
        if action_masks is not None:
            invalid_mask = (action_masks < 0.5)
            policy_logits = policy_logits.masked_fill(invalid_mask, float('-inf'))
        
        # Distribution actuelle (stabilité numérique)
        policy_logits = policy_logits.clamp(-50, 50)
        dist = Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Ratio de probabilité (clamper pour éviter explosion)
        ratio = torch.exp((log_probs - old_log_probs).clamp(-20, 20))
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Loss totale
        loss = (
            policy_loss 
            + self.value_coef * value_loss 
            - self.entropy_coef * entropy
        )
        
        # Skip si la loss est NaN
        if torch.isnan(loss):
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Vérifier les gradients NaN — skip si corrompus
        has_nan_grad = False
        for p in self.network.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        if has_nan_grad:
            self.optimizer.zero_grad()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.max_grad_norm
        )
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
    
    def set_training_mode(self, training: bool) -> None:
        """Active ou désactive le mode entraînement."""
        self._training = training
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Retourne les probabilités d'action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, _ = self.network(state_tensor)
            if torch.isnan(policy_logits).any():
                policy_logits = torch.zeros_like(policy_logits)
            policy_logits = policy_logits.clamp(-50, 50)
            probs = torch.softmax(policy_logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            return probs.cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
            "config": self.get_config(),
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint.get("training_steps", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            "type": "PPO",
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test de PPO ===\n")
    
    state_dim = 4
    n_actions = 2
    
    # Créer l'agent
    agent = PPOAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dims=[32, 32],
        n_epochs=4,
        batch_size=8
    )
    print(f"Agent: {agent.name}")
    print(f"Config: {agent.get_config()}")
    
    # Simuler une collecte
    print("\nSimulation de trajectoire:")
    for step in range(20):
        state = np.random.randn(state_dim).astype(np.float32)
        action, log_prob, value = agent.act_with_value(state, training=True)
        reward = 1.0 if step == 19 else 0.0
        done = (step == 19)
        
        agent.store(state, action, reward, log_prob, value, done)
    
    print(f"Buffer size: {len(agent.buffer)}")
    
    # Mettre à jour
    metrics = agent.update(last_value=0.0)
    print(f"Métriques après update: {metrics}")
    
    # Test des probabilités
    probs = agent.get_action_probs(np.random.randn(state_dim))
    print(f"Probabilités: {probs}")
    
    print("\n[OK] Tests passes!")
