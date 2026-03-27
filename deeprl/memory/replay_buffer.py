"""
Replay Buffer - Mémoire pour l'Experience Replay.

L'Experience Replay est une technique fondamentale en Deep RL qui:
1. Stocke les transitions (s, a, r, s', done) dans un buffer
2. Échantillonne aléatoirement des mini-batches pour l'entraînement

Pourquoi c'est important:
- Casse la corrélation temporelle entre les échantillons consécutifs
- Permet de réutiliser les expériences plusieurs fois (efficacité des données)
- Stabilise l'entraînement des réseaux de neurones

Histoire:
- Introduit dans le papier DQN d'Atari (Mnih et al., 2015)
- C'est l'une des clés du succès de DQN

Variantes:
- Uniform Replay: échantillonnage uniforme (ce fichier)
- Prioritized Replay: priorité aux transitions importantes (prioritized_buffer.py)
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from collections import deque
import random


class Transition(NamedTuple):
    """
    Une transition (s, a, r, s', done).
    
    Représente une étape dans l'environnement.
    
    Attributes:
        state: État avant l'action
        action: Action effectuée
        reward: Récompense reçue
        next_state: État après l'action
        done: True si l'épisode est terminé
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Buffer de replay uniforme avec échantillonnage aléatoire.
    
    Stocke les transitions dans un buffer circulaire (FIFO).
    Les vieilles transitions sont supprimées quand le buffer est plein.
    
    Attributes:
        capacity: Taille maximale du buffer
        buffer: Stockage des transitions
        
    Exemple d'utilisation:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
        >>> states, actions, rewards, next_states, dones = batch
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Crée un replay buffer.
        
        Args:
            capacity: Nombre maximum de transitions à stocker
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
        # Pour la reproductibilité (optionnel)
        self._rng = random.Random()
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Ajoute une transition au buffer.
        
        Args:
            state: État avant l'action
            action: Action effectuée
            reward: Récompense reçue
            next_state: État après l'action
            done: True si l'épisode est terminé
        """
        transition = Transition(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.array(next_state, dtype=np.float32),
            done=done
        )
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Échantillonne un mini-batch de transitions.
        
        Args:
            batch_size: Nombre de transitions à échantillonner
        
        Returns:
            Tuple (states, actions, rewards, next_states, dones)
            Chaque élément est un numpy array de forme (batch_size, ...)
        
        Raises:
            ValueError: Si batch_size > len(buffer)
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size ({batch_size}) > buffer size ({len(self.buffer)})"
            )
        
        # Échantillonner aléatoirement
        transitions = self._rng.sample(list(self.buffer), batch_size)
        
        # Séparer les composants et les empiler
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """
        Échantillonne des transitions (format Transition).
        
        Args:
            batch_size: Nombre de transitions
        
        Returns:
            Liste de Transition
        """
        return self._rng.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        """Retourne le nombre de transitions stockées."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Vérifie si le buffer a assez de transitions.
        
        Args:
            batch_size: Taille de batch souhaitée
        
        Returns:
            True si on peut échantillonner un batch de cette taille
        """
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Vide le buffer."""
        self.buffer.clear()
    
    def set_seed(self, seed: int) -> None:
        """Définit la graine aléatoire pour la reproductibilité."""
        self._rng.seed(seed)
    
    def get_stats(self) -> dict:
        """
        Retourne des statistiques sur le buffer.
        
        Returns:
            Dictionnaire avec les stats
        """
        if len(self.buffer) == 0:
            return {"size": 0, "capacity": self.capacity}
        
        rewards = [t.reward for t in self.buffer]
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fill_ratio": len(self.buffer) / self.capacity,
            "mean_reward": np.mean(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }
    
    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self.buffer)}, capacity={self.capacity})"


class EpisodeBuffer:
    """
    Buffer qui stocke des épisodes complets.
    
    Utile pour les algorithmes comme REINFORCE qui ont besoin
    de trajectoires complètes pour calculer les returns.
    """
    
    def __init__(self):
        """Crée un buffer d'épisode."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []  # Pour policy gradient
        self.values: List[float] = []  # Pour actor-critic
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: Optional[float] = None,
        value: Optional[float] = None
    ) -> None:
        """
        Ajoute une transition à l'épisode courant.
        
        Args:
            state: État
            action: Action
            reward: Récompense
            log_prob: Log-probabilité de l'action (pour REINFORCE)
            value: Valeur estimée de l'état (pour A2C)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
    
    def get_returns(self, gamma: float = 0.99) -> np.ndarray:
        """
        Calcule les returns (récompenses cumulées actualisées).
        
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        
        Args:
            gamma: Facteur d'actualisation
        
        Returns:
            Array des returns pour chaque timestep
        """
        returns = []
        G = 0
        
        # Parcourir les récompenses à l'envers
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return np.array(returns, dtype=np.float32)
    
    def get_advantages(self, gamma: float = 0.99) -> np.ndarray:
        """
        Calcule les avantages A_t = G_t - V(s_t).
        
        Nécessite que les valeurs aient été stockées.
        
        Args:
            gamma: Facteur d'actualisation
        
        Returns:
            Array des avantages
        """
        returns = self.get_returns(gamma)
        values = np.array(self.values)
        return returns - values
    
    def get_batch(self) -> Tuple[np.ndarray, ...]:
        """
        Retourne toutes les données de l'épisode.
        
        Returns:
            Tuple (states, actions, rewards, returns)
        """
        returns = self.get_returns()
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            returns
        )
    
    def clear(self) -> None:
        """Vide le buffer pour un nouvel épisode."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self) -> int:
        return len(self.rewards)


# Test
if __name__ == "__main__":
    print("=== Test du Replay Buffer ===\n")
    
    # Test 1: Buffer basique
    print("1. Test ReplayBuffer:")
    buffer = ReplayBuffer(capacity=100)
    
    # Ajouter des transitions
    for i in range(50):
        state = np.random.randn(4)
        next_state = np.random.randn(4)
        buffer.push(state, action=i % 2, reward=i * 0.1, 
                   next_state=next_state, done=(i == 49))
    
    print(f"   {buffer}")
    print(f"   Stats: {buffer.get_stats()}")
    
    # Échantillonner
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"   Batch shapes: states={states.shape}, actions={actions.shape}")
    
    # Test 2: Vérifier le comportement circulaire
    print("\n2. Test buffer circulaire:")
    buffer = ReplayBuffer(capacity=10)
    for i in range(15):
        buffer.push(np.array([i]), 0, i, np.array([i+1]), False)
    
    print(f"   Taille après 15 pushes dans buffer(10): {len(buffer)}")
    
    # Test 3: EpisodeBuffer
    print("\n3. Test EpisodeBuffer:")
    ep_buffer = EpisodeBuffer()
    
    # Simuler un épisode
    for i in range(5):
        ep_buffer.push(
            state=np.array([i]),
            action=i,
            reward=1.0 if i == 4 else 0.0
        )
    
    returns = ep_buffer.get_returns(gamma=0.99)
    print(f"   Rewards: {ep_buffer.rewards}")
    print(f"   Returns (γ=0.99): {returns}")
    
    print("\n[OK] Tests passes!")
