"""
REINFORCE - L'algorithme Policy Gradient fondamental.

REINFORCE (Williams, 1992) est l'algorithme de policy gradient le plus simple.
Il apprend directement une politique π(a|s) en suivant le gradient de la
performance attendue.

Intuition:
- Si une action mène à une bonne récompense, augmenter sa probabilité
- Si une action mène à une mauvaise récompense, diminuer sa probabilité

Équation du gradient:
    ∇J(θ) = E[∑_t ∇log π(a_t|s_t) * G_t]

où G_t est le return (récompense cumulée depuis t).

Variantes implémentées:
1. REINFORCE standard: utilise G_t directement
2. Avec baseline: G_t - b pour réduire la variance
3. Avec baseline appris (Critic): G_t - V(s_t)

La baseline réduit la variance sans introduire de biais.

Référence:
- "Simple Statistical Gradient-Following Algorithms" (Williams, 1992)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Optional, Dict, Any

from deeprl.agents.base import Agent
from deeprl.networks.mlp import MLP, ActorCriticMLP
from deeprl.memory.replay_buffer import EpisodeBuffer


class REINFORCEAgent(Agent):
    """
    Agent REINFORCE (Policy Gradient).
    
    Apprend une politique π(a|s) directement en maximisant
    la récompense attendue.
    
    Variantes:
    - baseline="none": REINFORCE standard
    - baseline="mean": Soustrait la moyenne des returns
    - baseline="critic": Utilise un réseau Critic pour V(s)
    
    Exemple d'utilisation:
        >>> agent = REINFORCEAgent(state_dim=4, n_actions=2)
        >>> action = agent.act(state)
        >>> agent.store_transition(state, action, reward, log_prob)
        >>> agent.learn_from_episode()  # À la fin de l'épisode
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        # Architecture
        hidden_dims: List[int] = [64, 64],
        # Hyperparamètres
        lr: float = 1e-3,
        gamma: float = 0.99,
        # Baseline
        baseline: str = "mean",  # "none", "mean", "critic"
        critic_lr: float = 1e-3,
        # Régularisation
        entropy_coef: float = 0.01,
        # Autres
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent REINFORCE.
        
        Args:
            state_dim: Dimension de l'état
            n_actions: Nombre d'actions
            hidden_dims: Architecture du réseau
            lr: Learning rate pour la politique
            gamma: Facteur d'actualisation
            baseline: Type de baseline ("none", "mean", "critic")
            critic_lr: Learning rate pour le critic (si baseline="critic")
            entropy_coef: Coefficient de régularisation par entropie
            device: "cpu" ou "cuda"
            seed: Graine aléatoire
        """
        name = "REINFORCE"
        if baseline == "mean":
            name += " (mean baseline)"
        elif baseline == "critic":
            name += " (Actor-Critic)"
        
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name=name,
            device=device
        )
        
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.gamma = gamma
        self.baseline = baseline
        self.entropy_coef = entropy_coef
        
        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Créer les réseaux
        if baseline == "critic":
            # Réseau Actor-Critic partagé
            self.network = ActorCriticMLP(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        else:
            # Réseau Policy seulement
            self.policy_network = MLP(
                state_dim=state_dim,
                output_dim=n_actions,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # Buffer pour stocker l'épisode courant
        self.episode_buffer = EpisodeBuffer()
        
        # Pour stocker les log-probs et entropies pendant l'épisode
        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_values: List[torch.Tensor] = []
        self.saved_entropies: List[torch.Tensor] = []
        
        # Running mean pour baseline='mean'
        self._running_return_mean = 0.0
        self._running_return_count = 0
        
        self._training = True
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action selon la politique apprise.
        
        En mode training, échantillonne selon π(a|s).
        En mode évaluation, peut soit échantillonner soit prendre argmax.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: Mode entraînement
        
        Returns:
            Action choisie
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Obtenir les logits de la politique
        if self.baseline == "critic":
            policy_logits, value = self.network(state_tensor)
        else:
            policy_logits = self.policy_network(state_tensor)
            value = None
        
        # Remplacer NaN par 0 (distribution uniforme) si le réseau diverge
        if torch.isnan(policy_logits).any():
            policy_logits = torch.zeros_like(policy_logits)
        
        # Masquer les actions non disponibles
        if available_actions is not None:
            mask = torch.full((self.n_actions,), float('-inf')).to(self.device)
            for a in available_actions:
                mask[a] = 0
            policy_logits = policy_logits + mask
        
        # Créer la distribution (stabilité numérique)
        policy_logits = policy_logits.clamp(-50, 50)
        probs = torch.softmax(policy_logits, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs)
        
        # Échantillonner ou prendre le max
        use_training = training and self._training
        if use_training:
            action = dist.sample()
        else:
            action = probs.argmax(dim=-1)
        
        # Sauvegarder pour l'apprentissage
        if use_training:
            log_prob = dist.log_prob(action)
            self.saved_log_probs.append(log_prob)
            self.saved_entropies.append(dist.entropy().squeeze())
            if value is not None:
                self.saved_values.append(value.squeeze())
        
        return int(action.item())
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float
    ) -> None:
        """
        Stocke une transition dans le buffer d'épisode.
        
        Note: log_prob est déjà stocké dans act() si training=True.
        """
        self.episode_buffer.push(state, action, reward)
    
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
        Interface standard - stocke la transition.
        
        L'apprentissage réel se fait à la fin de l'épisode.
        """
        self.store_transition(state, action, reward)
        
        if done:
            return self.learn_from_episode()
        return None
    
    def learn_from_episode(self) -> Dict[str, float]:
        """
        Effectue la mise à jour à la fin d'un épisode.
        
        REINFORCE nécessite l'épisode complet pour calculer les returns.
        
        Returns:
            Métriques d'apprentissage
        """
        if len(self.episode_buffer) == 0:
            return {}
        
        # Calculer les returns (non normalisés)
        returns_raw = self.episode_buffer.get_returns(self.gamma)
        returns = torch.FloatTensor(returns_raw).to(self.device)
        
        # Calculer la baseline et les avantages
        # IMPORTANT: Ne pas normaliser les returns au sein d'un seul épisode !
        # La normalisation fait que les premières actions (moins négatives) sont
        # renforcées même dans un épisode perdant.
        if self.baseline == "mean":
            # Baseline = running mean des returns des épisodes précédents
            # Cela permet de distinguer les bons épisodes des mauvais
            baseline = self._running_return_mean
            advantages = returns - baseline
            
            # Mettre à jour le running mean
            episode_return = returns[0].item()  # G_0 = return total de l'épisode
            self._running_return_count += 1
            self._running_return_mean += (episode_return - self._running_return_mean) / self._running_return_count
        elif self.baseline == "critic":
            # Pour actor-critic, on utilise V(s) comme baseline
            values = torch.stack(self.saved_values)
            advantages = returns - values.detach()
            
            # Loss du critic (MSE sur returns raw)
            critic_loss = nn.functional.mse_loss(values, returns)
        else:
            # Pas de baseline - utiliser les returns directement
            advantages = returns
        
        # Calculer la policy loss
        log_probs = torch.stack(self.saved_log_probs)
        
        # Policy gradient: -log π(a|s) * A(s, a)
        # Le moins car on fait gradient descent, pas ascent
        policy_loss = -(log_probs * advantages).mean()
        
        # Bonus d'entropie pour encourager l'exploration
        # L'entropie mesure l'incertitude de la politique: H(π) = -Σ π(a|s) log π(a|s)
        # On utilise l'entropie de la distribution complète (sauvée dans act())
        # et non une approximation à partir des seules log-probs échantillonnées.
        entropy = torch.stack(self.saved_entropies).mean()
        entropy_loss = -self.entropy_coef * entropy  # Négatif car on veut maximiser l'entropie
        
        # Loss totale
        if self.baseline == "critic":
            loss = policy_loss + 0.5 * critic_loss + entropy_loss
        else:
            loss = policy_loss + entropy_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters() if self.baseline == "critic" 
            else self.policy_network.parameters(),
            max_norm=0.5
        )
        self.optimizer.step()
        
        # Métriques
        metrics = {
            "policy_loss": policy_loss.item(),
            "mean_return": returns.mean().item(),
            "entropy": entropy.item(),
        }
        if self.baseline == "critic":
            metrics["critic_loss"] = critic_loss.item()
            metrics["mean_value"] = values.mean().item()
        
        # Nettoyer
        self.episode_buffer.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.saved_entropies.clear()
        
        self.training_steps += 1
        
        return metrics
    
    def on_episode_start(self) -> None:
        """Réinitialise le buffer au début de l'épisode."""
        self.episode_buffer.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.saved_entropies.clear()
    
    def set_training_mode(self, training: bool) -> None:
        """Active ou désactive le mode entraînement."""
        self._training = training
        if self.baseline == "critic":
            if training:
                self.network.train()
            else:
                self.network.eval()
        else:
            if training:
                self.policy_network.train()
            else:
                self.policy_network.eval()
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Retourne les probabilités d'action pour un état."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.baseline == "critic":
                logits, _ = self.network(state_tensor)
            else:
                logits = self.policy_network(state_tensor)
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
            logits = logits.clamp(-50, 50)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            return probs.cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        if self.baseline == "critic":
            state_dict = self.network.state_dict()
        else:
            state_dict = self.policy_network.state_dict()
        
        torch.save({
            "network": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
            "config": self.get_config(),
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if self.baseline == "critic":
            self.network.load_state_dict(checkpoint["network"])
        else:
            self.policy_network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint.get("training_steps", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            "type": self.name,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "gamma": self.gamma,
            "baseline": self.baseline,
            "entropy_coef": self.entropy_coef,
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test de REINFORCE ===\n")
    
    state_dim = 4
    n_actions = 2
    
    # Test 1: REINFORCE standard
    print("1. REINFORCE standard:")
    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        baseline="none"
    )
    print(f"   {agent.name}")
    
    # Test 2: Avec mean baseline
    print("\n2. REINFORCE avec mean baseline:")
    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        baseline="mean"
    )
    print(f"   {agent.name}")
    
    # Test 3: Actor-Critic
    print("\n3. REINFORCE Actor-Critic:")
    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        baseline="critic"
    )
    print(f"   {agent.name}")
    
    # Test 4: Simulation d'épisode
    print("\n4. Simulation d'épisode:")
    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        baseline="critic"
    )
    
    # Simuler un épisode
    for step in range(10):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.act(state, training=True)
        reward = 1.0 if step == 9 else 0.0
        done = (step == 9)
        
        agent.store_transition(state, action, reward)
    
    # Apprendre de l'épisode
    metrics = agent.learn_from_episode()
    print(f"   Métriques: {metrics}")
    
    # Test des probabilités
    probs = agent.get_action_probs(np.random.randn(state_dim))
    print(f"   Probabilités: {probs}")
    
    print("\n[OK] Tests passes!")
