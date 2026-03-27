"""
TicTacToe - Jeu à 2 joueurs classique.

Le Morpion est un excellent environnement pour:
- Tester les algorithmes multi-joueurs
- Apprendre contre un adversaire (Random, Heuristique, ou autre agent)
- Introduire les concepts de jeux à somme nulle

Règles:
- Grille 3x3
- 2 joueurs alternent (X et O)
- Aligner 3 symboles pour gagner
- Match nul si grille pleine sans gagnant

Représentation:
- État: grille 3x3 avec -1 (O), 0 (vide), +1 (X)
- Actions: 0-8 (positions sur la grille)
- Récompenses: +1 (victoire), -1 (défaite), 0 (nul/en cours)
"""

import numpy as np
from typing import List, Tuple, Optional
from deeprl.envs.base import Environment


class TicTacToe(Environment):
    """
    Environnement TicTacToe (Morpion) pour 2 joueurs.
    
    Le jeu alterne entre le joueur 0 (X, valeur +1) et 
    le joueur 1 (O, valeur -1).
    
    Actions (positions sur la grille):
        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
    
    États:
        Grille aplatie de 9 valeurs: -1 (O), 0 (vide), +1 (X)
        Ou one-hot encoding (9*3 = 27 dimensions)
    
    Récompenses (du point de vue du joueur courant):
        +1 : Victoire
        -1 : Défaite
        0  : Match nul ou partie en cours
    
    Exemple d'utilisation:
        >>> env = TicTacToe()
        >>> state = env.reset()
        >>> state, reward, done = env.step(4)  # Jouer au centre
    """
    
    # Joueurs
    PLAYER_X = 0  # Joue en premier, utilise +1 sur la grille
    PLAYER_O = 1  # Joue en second, utilise -1 sur la grille
    
    # Valeurs sur la grille
    EMPTY = 0
    X_VAL = 1
    O_VAL = -1
    
    # Lignes gagnantes (indices)
    WINNING_LINES = [
        [0, 1, 2],  # Ligne du haut
        [3, 4, 5],  # Ligne du milieu
        [6, 7, 8],  # Ligne du bas
        [0, 3, 6],  # Colonne gauche
        [1, 4, 7],  # Colonne milieu
        [2, 5, 8],  # Colonne droite
        [0, 4, 8],  # Diagonale \
        [2, 4, 6],  # Diagonale /
    ]
    
    def __init__(self, use_onehot: bool = True):
        """
        Crée un environnement TicTacToe.
        
        Args:
            use_onehot: Si True, état en one-hot (27 dims), sinon raw (9 dims)
        """
        super().__init__(name="TicTacToe")
        
        self.use_onehot = use_onehot
        
        # Grille de jeu (3x3 aplatie en vecteur de 9)
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = self.PLAYER_X
        self._winner: Optional[int] = None  # 0, 1, ou None
        
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """
        Forme de l'état.
        
        - One-hot: (27,) - 9 cases × 3 valeurs possibles
        - Raw: (9,) - valeurs directes (-1, 0, +1)
        """
        if self.use_onehot:
            return (27,)
        else:
            return (9,)
    
    @property
    def n_actions(self) -> int:
        """9 actions possibles (une par case)."""
        return 9
    
    @property
    def current_player(self) -> int:
        """Retourne le joueur courant (0=X, 1=O)."""
        return self._current_player
    
    def reset(self) -> np.ndarray:
        """
        Réinitialise le jeu.
        
        Returns:
            État initial (grille vide)
        """
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = self.PLAYER_X
        self._winner = None
        self._done = False
        self._state = self._get_state()
        return self._state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Joue un coup.
        
        Args:
            action: Position (0-8) où placer le symbole
        
        Returns:
            (next_state, reward, done)
            
        Note:
            La récompense est du point de vue du joueur qui vient de jouer.
        """
        if self._done:
            raise RuntimeError("La partie est terminée. Appelez reset().")
        
        if action not in self.get_available_actions():
            raise ValueError(f"Action {action} invalide (case occupée ou hors limites)")
        
        # Déterminer la valeur à placer
        value = self.X_VAL if self._current_player == self.PLAYER_X else self.O_VAL
        
        # Placer le symbole
        self._board[action] = value
        
        # Vérifier victoire/nul
        reward = 0.0
        
        if self._check_winner(value):
            self._winner = self._current_player
            reward = 1.0  # Victoire pour le joueur courant
            self._done = True
        elif len(self.get_available_actions()) == 0:
            # Match nul (grille pleine sans gagnant)
            reward = 0.0
            self._done = True
        else:
            # Partie continue, changer de joueur
            self._current_player = 1 - self._current_player
        
        self._state = self._get_state()
        return self._state.copy(), reward, self._done
    
    def _check_winner(self, value: int) -> bool:
        """Vérifie si le joueur avec cette valeur a gagné."""
        for line in self.WINNING_LINES:
            if all(self._board[i] == value for i in line):
                return True
        return False
    
    def get_available_actions(self) -> List[int]:
        """Retourne les cases vides."""
        if self._done:
            return []
        return [i for i in range(9) if self._board[i] == self.EMPTY]
    
    def _get_state(self) -> np.ndarray:
        """Construit la représentation de l'état."""
        if self.use_onehot:
            return self._get_onehot_state()
        else:
            return self._get_raw_state()
    
    def _get_raw_state(self) -> np.ndarray:
        """État brut: valeurs directes de la grille."""
        return self._board.astype(np.float32)
    
    def _get_onehot_state(self) -> np.ndarray:
        """
        État one-hot: 3 canaux par case.
        
        Pour chaque case:
        - [1, 0, 0] si vide
        - [0, 1, 0] si X (+1)
        - [0, 0, 1] si O (-1)
        """
        state = np.zeros(27, dtype=np.float32)
        
        for i, val in enumerate(self._board):
            if val == self.EMPTY:
                state[i * 3] = 1.0
            elif val == self.X_VAL:
                state[i * 3 + 1] = 1.0
            else:  # O_VAL
                state[i * 3 + 2] = 1.0
        
        return state
    
    def get_board_2d(self) -> np.ndarray:
        """Retourne la grille sous forme 3x3."""
        return self._board.reshape(3, 3)
    
    def render(self, mode: str = "text") -> str:
        """
        Affiche le plateau de jeu.
        
        Exemple:
            X | O | X
            ---------
            O | X |  
            ---------
              |   | O
        """
        symbols = {self.EMPTY: " ", self.X_VAL: "X", self.O_VAL: "O"}
        
        lines = []
        for row in range(3):
            row_str = " | ".join(symbols[self._board[row * 3 + col]] for col in range(3))
            lines.append(f" {row_str} ")
            if row < 2:
                lines.append("-----------")
        
        # Ajouter info sur le joueur courant
        if not self._done:
            player = "X" if self._current_player == self.PLAYER_X else "O"
            lines.append(f"\nTour: Joueur {player}")
        else:
            if self._winner is not None:
                winner = "X" if self._winner == self.PLAYER_X else "O"
                lines.append(f"\nVictoire: Joueur {winner}!")
            else:
                lines.append("\nMatch nul!")
        
        output = "\n".join(lines)
        
        if mode == "text":
            print(output)
        
        return output
    
    def get_winner(self) -> Optional[int]:
        """Retourne le gagnant (0=X, 1=O) ou None."""
        return self._winner
    
    def get_symmetries(self, state: np.ndarray, action: int) -> List[Tuple[np.ndarray, int]]:
        """
        Retourne les symétries de l'état et de l'action.
        
        Le TicTacToe a 8 symétries (4 rotations × 2 réflexions).
        Utile pour l'augmentation de données.
        
        Args:
            state: État actuel
            action: Action associée
        
        Returns:
            Liste de (état_symétrique, action_symétrique)
        """
        # Conversion position <-> (row, col)
        def pos_to_rc(pos):
            return pos // 3, pos % 3
        
        def rc_to_pos(r, c):
            return r * 3 + c
        
        symmetries = []
        board = self._board.reshape(3, 3)
        
        for k in range(4):  # 4 rotations
            for flip in [False, True]:  # Avec/sans réflexion
                b = np.rot90(board, k)
                if flip:
                    b = np.fliplr(b)
                
                # Transformer l'action
                r, c = pos_to_rc(action)
                # Appliquer les mêmes transformations
                for _ in range(k):
                    r, c = c, 2 - r  # Rotation 90°
                if flip:
                    c = 2 - c
                
                new_action = rc_to_pos(r, c)
                
                # Reconstruire l'état one-hot si nécessaire
                if self.use_onehot:
                    new_state = np.zeros(27, dtype=np.float32)
                    flat = b.flatten()
                    for i, val in enumerate(flat):
                        if val == self.EMPTY:
                            new_state[i * 3] = 1.0
                        elif val == self.X_VAL:
                            new_state[i * 3 + 1] = 1.0
                        else:
                            new_state[i * 3 + 2] = 1.0
                else:
                    new_state = b.flatten().astype(np.float32)
                
                symmetries.append((new_state, new_action))
        
        return symmetries
    
    def __repr__(self) -> str:
        return f"TicTacToe(player={self._current_player}, done={self._done})"


class TicTacToeVsRandom(TicTacToe):
    """
    TicTacToe où l'adversaire joue aléatoirement.
    
    L'agent contrôle toujours le joueur X (premier à jouer).
    Après chaque coup de l'agent, l'adversaire Random joue automatiquement.
    
    Cela simplifie l'entraînement: l'agent ne voit que ses propres tours.
    """
    
    def __init__(self, use_onehot: bool = True, seed: Optional[int] = None):
        """
        Args:
            use_onehot: Utiliser encoding one-hot
            seed: Graine pour l'adversaire aléatoire
        """
        super().__init__(use_onehot=use_onehot)
        self.name = "TicTacToeVsRandom"
        self.rng = np.random.default_rng(seed)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Joue le coup de l'agent, puis l'adversaire joue.
        
        Args:
            action: Action de l'agent (joueur X)
        
        Returns:
            (state, reward, done) après les deux coups
            
        Récompenses:
            +1 : Agent gagne
            -1 : Agent perd
            0  : Nul ou en cours
        """
        # L'agent joue (joueur X)
        state, reward, done = super().step(action)
        
        if done:
            return state, reward, done
        
        # L'adversaire Random joue (joueur O)
        available = self.get_available_actions()
        if len(available) > 0:
            opponent_action = self.rng.choice(available)
            state, opponent_reward, done = super().step(opponent_action)
            
            # Si l'adversaire gagne, c'est une défaite pour l'agent
            if done and opponent_reward == 1.0:
                reward = -1.0
        
        return state, reward, done
    
    @property
    def current_player(self) -> int:
        """L'agent est toujours le joueur 0 (X)."""
        return 0


# Test rapide
if __name__ == "__main__":
    print("=== Test de TicTacToe ===\n")
    
    # Test 1: Partie manuelle
    print("1. Partie manuelle:")
    env = TicTacToe()
    env.reset()
    env.render()
    
    # X joue au centre
    env.step(4)
    env.render()
    
    # O joue en haut à gauche
    env.step(0)
    env.render()
    
    # Test 2: Partie contre Random
    print("\n2. Partie contre Random:")
    env = TicTacToeVsRandom(seed=42)
    state = env.reset()
    print(f"State shape: {state.shape}")
    env.render()
    
    total_reward = 0
    while not env.is_game_over:
        available = env.get_available_actions()
        if len(available) == 0:
            break
        action = np.random.choice(available)
        state, reward, done = env.step(action)
        total_reward += reward
        print(f"\nAgent joue position {action}:")
        env.render()
    
    print(f"\nRécompense totale: {total_reward}")
    
    # Test 3: Stats sur plusieurs parties
    print("\n3. Stats sur 1000 parties (Random vs Random):")
    env = TicTacToeVsRandom()
    wins, losses, draws = 0, 0, 0
    
    for _ in range(1000):
        env.reset()
        while not env.is_game_over:
            available = env.get_available_actions()
            if len(available) == 0:
                break
            action = np.random.choice(available)
            _, reward, done = env.step(action)
            
            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
    
    print(f"   Victoires: {wins} ({wins/10:.1f}%)")
    print(f"   Défaites: {losses} ({losses/10:.1f}%)")
    print(f"   Nuls: {draws} ({draws/10:.1f}%)")
    
    print("\n[OK] Tests passes!")
