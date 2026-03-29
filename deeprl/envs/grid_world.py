"""
GridWorld - Environnement 2D pour le Reinforcement Learning.

Grille carrée où l'agent doit atteindre le goal en évitant l'état d'échec.

Configuration:
    - Départ: coin supérieur gauche (0, 0)
    - Goal: coin inférieur droit → récompense +1.0
    - Fail: coin supérieur droit → récompense -3.0
    - Déplacement: récompense 0.0

Schéma (grille 5x5):
    +---+---+---+---+---+
    | S |   |   |   | F |  ← Échec (-3.0)
    +---+---+---+---+---+
    |   |   |   |   |   |
    +---+---+---+---+---+
    |   |   |   |   |   |
    +---+---+---+---+---+
    |   |   |   |   |   |
    +---+---+---+---+---+
    |   |   |   |   | G |  ← Succès (+1.0)
    +---+---+---+---+---+
"""

import numpy as np
from typing import List, Tuple
from deeprl.envs.base import Environment


class GridWorld(Environment):
    """
    Environnement GridWorld - Grille 2D avec deux états terminaux.
    
    Actions:
        0 = Haut (↑), 1 = Bas (↓), 2 = Gauche (←), 3 = Droite (→)
    
    État:
        Vecteur one-hot de taille (size * size,) représentant la position.
    
    Exemple:
        >>> env = GridWorld(size=5)
        >>> state = env.reset()
        >>> next_state, reward, done = env.step(3)  # Droite
    """
    
    # Actions
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    # Déplacements (delta_row, delta_col)
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    
    # Récompenses
    REWARD_GOAL = 1.0
    REWARD_FAIL = -3.0
    REWARD_STEP = 0.0
    
    def __init__(self, size: int = 5):
        """
        Crée un GridWorld.
        
        Args:
            size: Taille de la grille carrée (défaut: 5)
        """
        super().__init__(name="GridWorld")
        self.size = size
        
        # Positions fixes
        self.start_pos = (0, 0)                      # Coin supérieur gauche
        self.goal_pos = (size - 1, size - 1)         # Coin inférieur droit
        self.fail_pos = (0, size - 1)                # Coin supérieur droit
        
        # État courant
        self._agent_pos = self.start_pos
        self._done = False
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Taille de l'état: vecteur one-hot de la position."""
        return (self.size * self.size,)
    
    @property
    def n_actions(self) -> int:
        """4 actions possibles."""
        return 4
    
    def reset(self) -> np.ndarray:
        """Réinitialise l'agent au départ."""
        self._agent_pos = self.start_pos
        self._done = False
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Exécute une action.
        
        Args:
            action: 0=haut, 1=bas, 2=gauche, 3=droite
        
        Returns:
            (état, récompense, terminé)
        """
        if self._done:
            raise RuntimeError("Épisode terminé. Appelez reset().")
        
        # Calculer nouvelle position
        dr, dc = self.MOVES[action]
        new_row = self._agent_pos[0] + dr
        new_col = self._agent_pos[1] + dc
        
        # Vérifier les limites
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self._agent_pos = (new_row, new_col)
        
        # Déterminer récompense et fin d'épisode
        if self._agent_pos == self.goal_pos:
            reward, self._done = self.REWARD_GOAL, True
        elif self._agent_pos == self.fail_pos:
            reward, self._done = self.REWARD_FAIL, True
        else:
            reward = self.REWARD_STEP
        
        return self._get_state(), reward, self._done
    
    def get_available_actions(self) -> List[int]:
        """Retourne toutes les actions (0-3)."""
        return [0, 1, 2, 3]
    
    def _get_state(self) -> np.ndarray:
        """Construit le vecteur one-hot de la position."""
        state = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self._agent_pos[0] * self.size + self._agent_pos[1]
        state[idx] = 1.0
        return state
    
    def render(self) -> str:
        """Affiche la grille."""
        lines = ["+" + "---+" * self.size]
        
        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                pos = (row, col)
                if pos == self._agent_pos:
                    cell = " A "
                elif pos == self.goal_pos:
                    cell = " G "
                elif pos == self.fail_pos:
                    cell = " F "
                else:
                    cell = " . "
                line += cell + "|"
            lines.append(line)
            lines.append("+" + "---+" * self.size)
        
        output = "\n".join(lines)
        print(output)
        return output
    
    def clone(self) -> "GridWorld":
        """Crée une copie de l'environnement."""
        env = GridWorld(self.size)
        env._agent_pos = self._agent_pos
        env._done = self._done
        return env
    
    # ------------------------------------------------------------------
    # Propriétés et méthodes utilitaires
    # ------------------------------------------------------------------

    @classmethod
    def create_simple(cls, size: int = 5) -> "GridWorld":
        """
        Crée un GridWorld simple (grille carrée classique).

        Args:
            size: Taille de la grille

        Returns:
            Instance de GridWorld
        """
        return cls(size=size)

    @property
    def width(self) -> int:
        """Largeur de la grille."""
        return self.size

    @property
    def height(self) -> int:
        """Hauteur de la grille."""
        return self.size

    @property
    def walls(self) -> set:
        """Ensemble des positions de murs (vide par défaut)."""
        return set()

    def pos_to_index(self, pos: Tuple[int, int]) -> int:
        """
        Convertit une position (row, col) en index linéaire.

        Args:
            pos: Tuple (row, col)

        Returns:
            Index dans le vecteur d'état one-hot
        """
        return pos[0] * self.size + pos[1]

    def get_optimal_path_length(self) -> int:
        """
        Longueur du chemin optimal (distance de Manhattan start → goal).

        Returns:
            Nombre minimum de pas pour atteindre le goal
        """
        return (abs(self.goal_pos[0] - self.start_pos[0])
                + abs(self.goal_pos[1] - self.start_pos[1]))

    def __repr__(self) -> str:
        return f"GridWorld(size={self.size}, pos={self._agent_pos})"


if __name__ == "__main__":
    print("=== Test GridWorld ===\n")
    
    env = GridWorld(size=5)
    env.reset()
    env.render()
    
    print("\nNavigation vers le goal (droite x4, bas x4):")
    actions = [3, 3, 3, 3, 1, 1, 1, 1]  # 4 droite + 4 bas
    
    for action in actions:
        state, reward, done = env.step(action)
        print(f"Action: {env.ACTION_NAMES[action]}, Reward: {reward:+.1f}")
        if done:
            env.render()
            print("OK Épisode terminé!")
            break
