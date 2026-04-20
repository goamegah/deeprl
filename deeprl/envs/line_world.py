"""
LineWorld - Un environnement 1D simple pour tester les algorithmes.

Un monde linéaire simple où l'agent peut se déplacer à gauche ou à droite.
L'objectif est d'atteindre l'extrémité droite tout en évitant l'extrémité gauche.

Paramètres du constructeur:
    size: Longueur de la ligne (par défaut: 5)

Caractéristiques:
    - États: size
    - Actions: 2 (gauche, droite)
    - Récompenses: -1.0 (échec), 0.0 (déplacement), +1.0 (succès)
    - États terminaux:
        * extrémité gauche (position 0) : récompense négative -1.0
        * extrémité droite (position size-1) : récompense positive +1.0

Schéma (size=5):
    [F][ ][S][ ][G]
     0  1  2  3  4
    
    S = Start (position initiale au milieu)
    G = Goal (succès, +1.0)
    F = Fail (échec, -1.0)
"""

import numpy as np
from typing import List, Tuple
from deeprl.envs.base import Environment


class LineWorld(Environment):
    """
    Environnement LineWorld - Une ligne 1D avec deux états terminaux.
    
    L'agent commence au milieu et doit atteindre l'extrémité droite (succès)
    tout en évitant l'extrémité gauche (échec).
    
    Actions:
        0 = Gauche
        1 = Droite
    
    Récompenses:
        +1.0 : Atteindre l'extrémité droite (succès)
        -1.0 : Atteindre l'extrémité gauche (échec)
         0.0 : Chaque déplacement
    
    États terminaux:
        - Position 0 (extrémité gauche) : reward = -1.0
        - Position size-1 (extrémité droite) : reward = +1.0
    
    Exemple d'utilisation:
        >>> env = LineWorld(size=5)
        >>> state = env.reset()  # Démarre au milieu
        >>> next_state, reward, done = env.step(1)  # Aller à droite
    """
    
    # Actions possibles
    LEFT = 0
    RIGHT = 1
    
    # Récompenses
    REWARD_SUCCESS = 1.0   # Atteindre l'extrémité droite
    REWARD_FAILURE = -1.0  # Atteindre l'extrémité gauche
    REWARD_STEP = 0.0      # Déplacement normal
    
    def __init__(self, size: int = 5):
        """
        Crée un environnement LineWorld.
        
        Args:
            size: Longueur de la ligne (>= 3 pour avoir un milieu)
        """
        super().__init__(name="LineWorld")
        
        if size < 3:
            raise ValueError("La taille doit être au moins 3")
        
        self.size = size
        self._start_position = size // 2  # Position initiale au milieu
        self._position = self._start_position
        
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """
        L'état est un vecteur one-hot de taille `size`.
        
        Exemple pour size=5, position=2:
            [0, 0, 1, 0, 0]
        """
        return (self.size,)
    
    @property
    def n_actions(self) -> int:
        """2 actions: gauche (0) et droite (1)."""
        return 2
    
    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement.
        
        L'agent recommence au milieu de la ligne.
        
        Returns:
            État initial (one-hot encoding de la position centrale)
        """
        self._position = self._start_position
        self._done = False
        return self.get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: 0 (gauche) ou 1 (droite)
        
        Returns:
            (next_state, reward, done)
        """
        if self._done:
            raise RuntimeError("L'épisode est terminé. Appelez reset().")
        
        if action not in self.get_available_actions():
            raise ValueError(f"Action {action} invalide. Actions valides: {self.get_available_actions()}")
        
        # Mise à jour de la position
        if action == self.LEFT:
            self._position = max(0, self._position - 1)
        elif action == self.RIGHT:
            self._position = min(self.size - 1, self._position + 1)
        
        # Calculer la récompense
        reward = self.REWARD_STEP  # 0.0 pour déplacement normal
        
        # Vérifier les états terminaux
        if self._position == self.size - 1:  # Extrémité droite = succès
            reward = self.REWARD_SUCCESS  # +1.0
            self._done = True
        elif self._position == 0:  # Extrémité gauche = échec
            reward = self.REWARD_FAILURE  # -1.0
            self._done = True
        
        return self.get_state(), reward, self._done
    
    def get_available_actions(self) -> List[int]:
        """
        Toutes les actions sont toujours disponibles dans LineWorld.
        
        Même si l'agent est au bord, il peut essayer d'aller dans le mur
        (il restera juste sur place).
        """
        return [self.LEFT, self.RIGHT]
    
    def get_state(self) -> np.ndarray:
        """Vecteur one-hot de la position courante."""
        state = np.zeros(self.size, dtype=np.float32)
        state[self._position] = 1.0
        return state
    
    def render(self, mode: str = "text") -> str:
        """
        Affiche l'environnement en mode texte.
        
        Exemple de sortie pour position=2, size=5:
            [F][ ][A][ ][G]
             0  1  2  3  4
        
        A = Agent
        G = Goal (succès, +1.0)
        F = Fail (échec, -1.0)
        """
        cells = []
        for i in range(self.size):
            if i == self._position:
                cells.append("[A]")  # Agent
            elif i == self.size - 1:
                cells.append("[G]")  # Goal (succès)
            elif i == 0:
                cells.append("[F]")  # Fail (échec)
            else:
                cells.append("[ ]")  # Vide
        
        line1 = "".join(cells)
        line2 = " " + "  ".join(str(i) for i in range(self.size))
        
        output = f"\n{line1}\n{line2}\n"
        
        if mode == "text":
            print(output)
        
        return output
    
    def clone(self) -> "LineWorld":
        """Clone l'environnement."""
        env = LineWorld(self.size)
        env._position = self._position
        env._done = self._done
        return env
    
    def determinize(self, obs: np.ndarray) -> "LineWorld":
        """Reconstruit un LineWorld jouable à partir d'une observation one-hot."""
        env = LineWorld(self.size)
        env._position = int(np.argmax(obs))
        env._done = (env._position == 0 or env._position == self.size - 1)
        return env
    
    def __repr__(self) -> str:
        return f"LineWorld(size={self.size}, position={self._position})"


# Test rapide si exécuté directement
if __name__ == "__main__":
    print("=== Test de LineWorld ===\n")
    
    # Créer l'environnement
    env = LineWorld(size=5)
    print(f"Environnement créé: {env}")
    print(f"  - State shape: {env.state_shape}")
    print(f"  - Nombre d'actions: {env.n_actions}")
    
    # Reset et afficher l'état initial
    state = env.reset()
    print(f"\nÉtat initial (one-hot): {state}")
    env.render()
    
    # Faire quelques pas
    print("\n--- Simulation: toujours aller à droite ---")
    total_reward = 0
    step_count = 0
    
    while not env.is_game_over:
        action = LineWorld.RIGHT
        next_state, reward, done = env.step(action)
        total_reward += reward
        step_count += 1
        
        action_name = "Droite" if action == 1 else "Gauche"
        print(f"Step {step_count}: Action={action_name}, Reward={reward:.2f}, Done={done}")
        env.render()
    
    print(f"\n[OK] Episode termine!")
    print(f"   Nombre de pas: {step_count}")
    print(f"   Recompense totale: {total_reward:.2f}")
