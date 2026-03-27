"""
Prioritized Experience Replay Buffer.

Amélioration du Replay Buffer standard qui échantillonne les transitions
selon leur "importance" (TD error).

Intuition:
- Les transitions avec une grande TD error sont "surprenantes"
- On apprend plus de ces transitions
- Donc on les échantillonne plus souvent

Deux types de priorité:
- Proportional: P(i) ∝ |δ_i|^α où δ_i est la TD error
- Rank-based: P(i) ∝ 1/rank(i)^α

Importance Sampling:
- L'échantillonnage biaisé nécessite une correction
- On utilise des poids w_i = (N * P(i))^{-β}
- β augmente de 0 à 1 pendant l'entraînement

Référence:
- "Prioritized Experience Replay" (Schaul et al., 2015)
"""

import numpy as np
from typing import Tuple, Optional
from deeprl.memory.replay_buffer import Transition


class SumTree:
    """
    Structure de données Sum Tree pour échantillonnage efficace O(log n).
    
    Un Sum Tree est un arbre binaire complet où:
    - Les feuilles contiennent les priorités
    - Chaque nœud parent contient la somme de ses enfants
    - La racine contient la somme totale
    
    Exemple pour 4 éléments avec priorités [0.1, 0.5, 0.3, 0.1]:
    
                    [1.0]           <- Somme totale
                   /     \
               [0.6]     [0.4]
               /   \     /   \
            [0.1] [0.5] [0.3] [0.1]  <- Priorités (feuilles)
    
    Opérations:
    - update(idx, priority): O(log n)
    - sample(): O(log n)
    - total_priority: O(1)
    """
    
    def __init__(self, capacity: int):
        """
        Crée un Sum Tree.
        
        Args:
            capacity: Nombre maximum d'éléments (feuilles)
        """
        self.capacity = capacity
        
        # L'arbre a 2*capacity - 1 nœuds
        # Les indices 0 à capacity-2 sont les nœuds internes
        # Les indices capacity-1 à 2*capacity-2 sont les feuilles
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        
        # Stockage des données (transitions)
        self.data = [None] * capacity
        
        # Pointeur pour l'insertion circulaire
        self.write_idx = 0
        self.size = 0
    
    def _leaf_idx_to_tree_idx(self, leaf_idx: int) -> int:
        """Convertit un indice de feuille en indice d'arbre."""
        return leaf_idx + self.capacity - 1
    
    def _propagate(self, tree_idx: int, change: float) -> None:
        """Propage un changement de priorité vers la racine."""
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def update(self, leaf_idx: int, priority: float) -> None:
        """
        Met à jour la priorité d'une feuille.
        
        Args:
            leaf_idx: Indice de la feuille (0 à capacity-1)
            priority: Nouvelle priorité
        """
        tree_idx = self._leaf_idx_to_tree_idx(leaf_idx)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def add(self, priority: float, data) -> int:
        """
        Ajoute un élément avec une priorité donnée.
        
        Args:
            priority: Priorité de l'élément
            data: Données à stocker (transition)
        
        Returns:
            Indice de la feuille où l'élément a été stocké
        """
        leaf_idx = self.write_idx
        self.data[leaf_idx] = data
        self.update(leaf_idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        return leaf_idx
    
    def get(self, value: float) -> Tuple[int, float, Transition]:
        """
        Trouve l'élément correspondant à une valeur cumulée.
        
        Args:
            value: Valeur dans [0, total_priority]
        
        Returns:
            (leaf_idx, priority, data)
        """
        tree_idx = 0  # Commencer à la racine
        
        while True:
            left = 2 * tree_idx + 1
            right = left + 1
            
            if left >= len(self.tree):
                # C'est une feuille
                break
            
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right
        
        leaf_idx = tree_idx - (self.capacity - 1)
        return leaf_idx, self.tree[tree_idx], self.data[leaf_idx]
    
    @property
    def total_priority(self) -> float:
        """Somme totale des priorités."""
        return self.tree[0]
    
    @property
    def min_priority(self) -> float:
        """Priorité minimale non-nulle."""
        leaf_priorities = self.tree[self.capacity - 1:self.capacity - 1 + self.size]
        non_zero = leaf_priorities[leaf_priorities > 0]
        return non_zero.min() if len(non_zero) > 0 else 1.0
    
    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Buffer de replay avec échantillonnage prioritaire.
    
    Utilise un Sum Tree pour un échantillonnage efficace O(log n).
    
    Hyperparamètres:
        α (alpha): Exposant de priorité (0 = uniforme, 1 = full prioritized)
        β (beta): Correction de biais (0 = pas de correction, 1 = correction totale)
        ε (epsilon): Petite constante pour éviter priorité = 0
    
    Exemple d'utilisation:
        >>> buffer = PrioritizedReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch, weights, indices = buffer.sample(32, beta=0.4)
        >>> # Après calcul de la loss, mettre à jour les priorités
        >>> buffer.update_priorities(indices, td_errors)
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Crée un buffer de replay prioritaire.
        
        Args:
            capacity: Taille maximale du buffer
            alpha: Exposant de priorité (0=uniforme, 1=full prioritized)
            beta_start: Beta initial pour importance sampling
            beta_end: Beta final
            beta_frames: Nombre de frames pour atteindre beta_end
            epsilon: Petite constante ajoutée aux priorités
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        
        # Priorité maximale (pour les nouvelles transitions)
        self.max_priority = 1.0
        
        # Compteur de frames pour le schedule de beta
        self.frame = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Ajoute une transition avec priorité maximale.
        
        Les nouvelles transitions reçoivent la priorité maximale
        pour s'assurer qu'elles seront échantillonnées au moins une fois.
        """
        transition = Transition(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.array(next_state, dtype=np.float32),
            done=done
        )
        
        # Priorité = max_priority^alpha
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Échantillonne un batch avec importance sampling weights.
        
        Args:
            batch_size: Taille du batch
            beta: Exposant d'importance sampling (None = schedule automatique)
        
        Returns:
            - batch: (states, actions, rewards, next_states, dones)
            - weights: Poids d'importance sampling normalisés
            - indices: Indices des transitions (pour update_priorities)
        """
        if beta is None:
            beta = self._get_beta()
        
        transitions = []
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        
        # Diviser la priorité totale en segments
        segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            # Échantillonner dans chaque segment
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            
            idx, priority, transition = self.tree.get(value)
            indices[i] = idx
            priorities[i] = priority
            transitions.append(transition)
        
        # Calculer les poids d'importance sampling
        # w_i = (N * P(i))^{-β} / max(w)
        sampling_probs = priorities / self.tree.total_priority
        weights = (len(self.tree) * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normaliser
        
        # Extraire les composants du batch
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        
        batch = (states, actions, rewards, next_states, dones)
        
        self.frame += 1
        
        return batch, weights.astype(np.float32), indices
    
    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        Met à jour les priorités après le calcul de la TD error.
        
        Args:
            indices: Indices des transitions (retournés par sample)
            td_errors: TD errors absolues pour ces transitions
        """
        for idx, td_error in zip(indices, td_errors):
            # Priorité brute (avant exponentiation)
            raw_priority = abs(td_error) + self.epsilon
            
            # Mettre à jour le max (stocke la priorité brute)
            self.max_priority = max(self.max_priority, raw_priority)
            
            # Priorité dans l'arbre = p^α
            tree_priority = raw_priority ** self.alpha
            self.tree.update(int(idx), tree_priority)
    
    def _get_beta(self) -> float:
        """Calcule beta selon le schedule linéaire."""
        fraction = min(self.frame / self.beta_frames, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        return len(self.tree)
    
    def is_ready(self, batch_size: int) -> bool:
        """Vérifie si on peut échantillonner un batch."""
        return len(self.tree) >= batch_size
    
    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(size={len(self)}, capacity={self.capacity}, "
            f"α={self.alpha}, β={self._get_beta():.3f})"
        )


# Test
if __name__ == "__main__":
    print("=== Test du Prioritized Replay Buffer ===\n")
    
    # Test SumTree
    print("1. Test SumTree:")
    tree = SumTree(capacity=4)
    
    # Ajouter des éléments avec différentes priorités
    tree.add(0.1, "A")
    tree.add(0.5, "B")
    tree.add(0.3, "C")
    tree.add(0.1, "D")
    
    print(f"   Total priority: {tree.total_priority}")
    print(f"   Tree: {tree.tree}")
    
    # Échantillonner
    samples = {"A": 0, "B": 0, "C": 0, "D": 0}
    for _ in range(1000):
        value = np.random.uniform(0, tree.total_priority)
        _, _, data = tree.get(value)
        samples[data] += 1
    
    print(f"   Échantillons (1000 tirages): {samples}")
    print(f"   B devrait être le plus fréquent (priorité 0.5)")
    
    # Test PrioritizedReplayBuffer
    print("\n2. Test PrioritizedReplayBuffer:")
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
    
    # Ajouter des transitions
    for i in range(50):
        state = np.random.randn(4)
        next_state = np.random.randn(4)
        buffer.push(state, i % 2, i * 0.1, next_state, (i == 49))
    
    print(f"   {buffer}")
    
    # Échantillonner
    batch, weights, indices = buffer.sample(16)
    states, actions, rewards, next_states, dones = batch
    
    print(f"   Batch shapes: states={states.shape}")
    print(f"   Weights: min={weights.min():.3f}, max={weights.max():.3f}")
    
    # Mettre à jour les priorités
    td_errors = np.random.uniform(0, 1, size=16)
    buffer.update_priorities(indices, td_errors)
    print(f"   Priorités mises à jour")
    
    print("\n[OK] Tests passes!")
