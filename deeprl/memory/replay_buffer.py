"""
Buffers d'experience pour le Deep Reinforcement Learning.

Deux implementations :

1. ReplayBuffer (Mnih et al., 2015)
   - Stocke les transitions (s, a, r, s', done)
   - Echantillonne des mini-batches uniformement
   - Casse les correlations temporelles entre transitions

2. PrioritizedReplayBuffer (Schaul et al., 2016)
   - Les transitions avec un TD-error eleve sont echantillonnees plus souvent
   - Utilise un SumTree pour un echantillonnage O(log n)
   - Corrige le biais via Importance Sampling (IS)

References :
- Mnih et al. (2015) "Human-level control through deep RL"
- Schaul et al. (2016) "Prioritized Experience Replay"
- Sutton & Barto (2018), Ch. 8.4 — Prioritized Sweeping
"""

import numpy as np
from typing import Tuple


# ============================================================================
# REPLAY BUFFER UNIFORME
# ============================================================================

class ReplayBuffer:
    """
    Buffer d'experience avec echantillonnage uniforme.

    Stocke les N dernieres transitions dans un buffer circulaire.
    A chaque mise a jour, un mini-batch est echantillonne uniformement.

    Avantages par rapport a l'apprentissage en ligne :
    - Reutilise chaque experience plusieurs fois (sample efficiency)
    - Reduit la correlation entre mises a jour successives
    - Lisse la distribution d'apprentissage

    Exemple :
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> if len(buffer) >= 32:
        ...     batch = buffer.sample(32)
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Taille maximale du buffer. Les transitions les plus
                      anciennes sont ecrasees quand le buffer est plein.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        """Ajoute une transition au buffer."""
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Echantillonne un mini-batch uniformement.

        Args:
            batch_size: Taille du batch

        Returns:
            (states, actions, rewards, next_states, dones)
            Chaque element est un numpy array de taille (batch_size, ...).
        """
        indices = np.random.choice(
            len(self.buffer), size=batch_size,
            replace=(len(self.buffer) < batch_size)
        )

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# SUM TREE (structure de donnees pour PER)
# ============================================================================

class SumTree:
    """
    Arbre binaire de somme pour echantillonnage proportionnel en O(log n).

    Chaque feuille stocke une priorite. Les noeuds internes stockent la
    somme de leurs enfants. Echantillonner revient a tirer un nombre
    uniforme dans [0, total] et descendre dans l'arbre.

    Structure :
        Indices 0..(capacity-2) : noeuds internes
        Indices (capacity-1)..(2*capacity-2) : feuilles

    Reference : Schaul et al. (2016), Appendix B.2.1
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    @property
    def total(self) -> float:
        """Somme totale des priorites."""
        return float(self.tree[0])

    def add(self, priority: float, data):
        """Ajoute une donnee avec sa priorite."""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self._update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float):
        """Met a jour une feuille et propage vers la racine."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Remonter vers la racine
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def update(self, tree_idx: int, priority: float):
        """Met a jour la priorite d'une feuille existante."""
        self._update(tree_idx, priority)

    def get(self, cumsum: float) -> Tuple[int, float, object]:
        """
        Descend dans l'arbre pour trouver la feuille correspondant
        a la somme cumulative donnee.

        Args:
            cumsum: Valeur dans [0, total]

        Returns:
            (tree_idx, priority, data)
        """
        idx = 0  # Racine
        while True:
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):
                # C'est une feuille
                break

            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ============================================================================
# PRIORITIZED REPLAY BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Buffer d'experience avec echantillonnage prioritaire.

    Les transitions avec un TD-error eleve (|delta| grand) sont
    echantillonnees plus frequemment, ce qui accelere l'apprentissage
    sur les experiences les plus informatives.

    Probabilite d'echantillonnage :
        P(i) = p_i^alpha / sum_k p_k^alpha
        ou p_i = |delta_i| + epsilon (TD-error + petite constante)

    Correction par Importance Sampling (IS) :
        w_i = (N * P(i))^{-beta}
        Les poids w_i sont normalises par max(w).
        beta augmente de beta_start a 1.0 au cours de l'entrainement.

    Quand alpha=0 : echantillonnage uniforme (equivalent a ReplayBuffer).
    Quand beta=1  : correction IS complete (pas de biais).

    Exemple :
        >>> buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32, beta=0.4)
        >>> states, actions, rewards, next_states, dones, indices, weights = batch
        >>> # ... calculer td_errors ...
        >>> buffer.update_priorities(indices, td_errors)

    Reference : Schaul et al. (2016), Sections 3.3-3.4
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Args:
            capacity: Taille maximale du buffer
            alpha: Exposant de priorite (0 = uniforme, 1 = full priority)
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-6  # Evite les priorites nulles
        self.max_priority = 1.0  # Priorite max observee

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        """
        Ajoute une transition avec la priorite maximale.

        Les nouvelles transitions recoivent la priorite maximale pour
        garantir qu'elles soient echantillonnees au moins une fois.
        """
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Echantillonne un mini-batch selon les priorites.

        Args:
            batch_size: Taille du batch
            beta: Exposant IS (0 = pas de correction, 1 = correction complete)

        Returns:
            (states, actions, rewards, next_states, dones, tree_indices, is_weights)
        """
        batch = []
        indices = []
        priorities = []

        # Stratified sampling : divise [0, total] en segments egaux
        total = self.tree.total
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = np.random.uniform(low, high)

            tree_idx, priority, data = self.tree.get(cumsum)

            # Protection contre les donnees None (buffer pas encore plein)
            if data is None:
                cumsum = np.random.uniform(0, total)
                tree_idx, priority, data = self.tree.get(cumsum)
            if data is None:
                continue

            batch.append(data)
            indices.append(tree_idx)
            priorities.append(max(priority, self.epsilon))

        if len(batch) < batch_size:
            # Completer si necessaire
            while len(batch) < batch_size:
                cumsum = np.random.uniform(0, total)
                tree_idx, priority, data = self.tree.get(cumsum)
                if data is not None:
                    batch.append(data)
                    indices.append(tree_idx)
                    priorities.append(max(priority, self.epsilon))

        states, actions, rewards, next_states, dones = zip(*batch)

        # Importance Sampling weights : w_i = (N * P(i))^{-beta}
        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / total
        weights = (self.tree.size * probs) ** (-beta)
        weights = weights / weights.max()  # Normaliser

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Met a jour les priorites apres le calcul des TD-errors.

        Args:
            indices: Indices dans le SumTree (retournes par sample)
            td_errors: TD-errors absolus pour chaque transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def __len__(self) -> int:
        return self.tree.size
