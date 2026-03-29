"""
Quarto - Jeu de stratégie à deux joueurs.

Quarto est un jeu abstrait où:
- 16 pièces avec 4 attributs binaires (grand/petit, clair/foncé, plein/creux, rond/carré)
- Plateau 4×4
- Victoire: aligner 4 pièces partageant AU MOINS un attribut commun
- Particularité: c'est l'adversaire qui choisit quelle pièce vous jouez!

Phases du jeu:
1. Joueur A choisit une pièce pour B
2. Joueur B place cette pièce et choisit une pièce pour A
3. Répéter jusqu'à victoire ou plateau plein (nul)

Représentation des pièces (4 bits):
- Bit 0: Taille (0=petit, 1=grand)
- Bit 1: Couleur (0=clair, 1=foncé)
- Bit 2: Forme (0=creux, 1=plein)
- Bit 3: Sommet (0=rond, 1=carré)

Exemples:
- 0b0000 (0): petit, clair, creux, rond
- 0b1111 (15): grand, foncé, plein, carré
"""

import numpy as np
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

from deeprl.envs.base import Environment


@dataclass
class QuartoPiece:
    """
    Une pièce de Quarto avec 4 attributs.
    """
    tall: bool      # Grand vs Petit
    dark: bool      # Foncé vs Clair
    solid: bool     # Plein vs Creux
    square: bool    # Carré vs Rond
    
    @classmethod
    def from_id(cls, piece_id: int) -> "QuartoPiece":
        """Crée une pièce à partir de son ID (0-15)."""
        return cls(
            tall=bool(piece_id & 1),
            dark=bool(piece_id & 2),
            solid=bool(piece_id & 4),
            square=bool(piece_id & 8)
        )
    
    @classmethod
    def all_pieces(cls) -> List["QuartoPiece"]:
        """Retourne la liste de toutes les 16 pièces."""
        return [cls.from_id(i) for i in range(16)]
    
    def to_id(self) -> int:
        """Retourne l'ID de la pièce."""
        return (
            int(self.tall) |
            (int(self.dark) << 1) |
            (int(self.solid) << 2) |
            (int(self.square) << 3)
        )
    
    def __repr__(self) -> str:
        """Représentation courte de la pièce."""
        chars = ""
        chars += "G" if self.tall else "p"
        chars += "F" if self.dark else "c"
        chars += "P" if self.solid else "x"
        chars += "C" if self.square else "r"
        return chars


class Quarto(Environment):
    """
    Environnement Quarto.
    
    Actions:
    - Phase "place": 0-15 (position sur le plateau)
    - Phase "give": 0-15 (pièce à donner à l'adversaire)
    
    Espace d'actions unifié de 32 actions :
    - Actions 0-15  : placer la pièce sur une position du plateau (phase "place")
    - Actions 16-31 : donner une pièce à l'adversaire (phase "give"),
                      action 16 = pièce 0, ..., action 31 = pièce 15
    
    Pendant la phase "place", les actions 16-31 sont masquées.
    Pendant la phase "give", les actions 0-15 sont masquées.
    Chaque sortie d'un réseau a toujours la même sémantique.
    
    Représentation de l'état (pour réseau de neurones):
    - Board: 16 positions × 17 valeurs (16 pièces + vide) = 272 dims
    - Ou simplifié: 16 positions × 5 channels (4 attributs + présence)
    - Pièce courante: 16 dims (one-hot)
    - Pièces disponibles: 16 dims (binaire)
    - Joueur courant: 2 dims
    Total simplifié: 16×5 + 16 + 16 + 2 = 114 dims
    """
    
    N_PIECES = 16
    BOARD_SIZE = 4
    N_POSITIONS = 16
    
    # Lignes gagnantes: indices des 4 positions
    WINNING_LINES = [
        # Lignes horizontales
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
        # Lignes verticales
        [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
        # Diagonales
        [0, 5, 10, 15], [3, 6, 9, 12]
    ]
    
    def __init__(
        self,
        seed: Optional[int] = None
    ):
        """
        Crée un environnement Quarto.
        
        État: représentation compacte (114 dims).
        Actions: 32 actions (0-15 place, 16-31 give).
        
        Args:
            seed: Graine aléatoire
        """
        super().__init__(name="Quarto")
        
        self._state_dim = 16 * 5 + 16 + 16 + 2  # 114
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Forme de l'espace d'états."""
        return (self._state_dim,)
    
    @property
    def n_actions(self) -> int:
        """32 actions: 0-15 = positions (place), 16-31 = pièces (give)."""
        return 32
    
    def reset(self) -> np.ndarray:
        """Réinitialise le jeu."""
        # Plateau: -1 = vide, 0-15 = pièce
        self._board = np.full(self.N_POSITIONS, -1, dtype=np.int8)
        
        # Pièces disponibles
        self._available_pieces: Set[int] = set(range(self.N_PIECES))
        
        # Pièce courante à placer (None = phase "give")
        self._current_piece: Optional[int] = None
        
        # Joueur courant
        self._current_player = 0
        
        # Phase: "give" ou "place"
        self._phase = "give"  # Le premier joueur donne une pièce
        
        # État du jeu
        self._winner: Optional[int] = None
        self._done = False
        self._move_count = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Retourne l'état du jeu (compact, 114D)."""
        return self._get_compact_state()
    
    def _get_compact_state(self) -> np.ndarray:
        """
        État compact: 114 dimensions.
        
        Structure:
        - Plateau: 16 positions × 5 channels (4 attributs + présence)
        - Pièce courante: 16 dims (one-hot, 0 si aucune)
        - Pièces disponibles: 16 dims (binaire)
        - Joueur courant: 2 dims (one-hot)
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0
        
        # Plateau (80 dims)
        for pos in range(self.N_POSITIONS):
            piece_id = self._board[pos]
            if piece_id >= 0:
                # Présence
                state[idx] = 1.0
                # Attributs (normalisés à 0/1)
                state[idx + 1] = float(piece_id & 1)  # tall
                state[idx + 2] = float((piece_id >> 1) & 1)  # dark
                state[idx + 3] = float((piece_id >> 2) & 1)  # solid
                state[idx + 4] = float((piece_id >> 3) & 1)  # square
            idx += 5
        
        # Pièce courante (16 dims)
        if self._current_piece is not None:
            state[idx + self._current_piece] = 1.0
        idx += 16
        
        # Pièces disponibles (16 dims)
        for p in self._available_pieces:
            state[idx + p] = 1.0
        idx += 16
        
        # Joueur courant (2 dims)
        state[idx + self._current_player] = 1.0
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Effectue une action.
        
        Args:
            action: 0-15 = position (phase place), 16-31 = pièce (phase give)
        
        Returns:
            (state, reward, done)
        """
        if self._done:
            return self.get_state(), 0.0, True
        
        if self._phase == "place":
            # Action 0-15 : placer la pièce courante
            pos = action
            if pos < 0 or pos >= self.N_POSITIONS or self._board[pos] >= 0:
                # Action invalide
                return self.get_state(), -1.0, False
            
            # Placer la pièce
            self._board[pos] = self._current_piece
            self._available_pieces.discard(self._current_piece)
            self._current_piece = None
            self._move_count += 1
            
            # Vérifier victoire
            if self._check_win():
                self._winner = self._current_player
                self._done = True
                return self.get_state(), 1.0, True
            
            # Vérifier nul
            if len(self._available_pieces) == 0:
                self._winner = -1  # Nul
                self._done = True
                return self.get_state(), 0.0, True
            
            # Passer à la phase "give"
            self._phase = "give"
            return self.get_state(), 0.0, False
        
        else:  # Phase "give"
            # Action 16-31 : choisir une pièce pour l'adversaire
            piece = action - self.N_POSITIONS  # 16->0, 17->1, ..., 31->15
            
            if piece < 0 or piece >= self.N_PIECES or piece not in self._available_pieces:
                # Action invalide
                return self.get_state(), -1.0, False
            
            # Donner la pièce
            self._current_piece = piece
            
            # Changer de joueur et passer à "place"
            self._current_player = 1 - self._current_player
            self._phase = "place"
            
            return self.get_state(), 0.0, False
    
    def _check_win(self) -> bool:
        """Vérifie si le joueur courant a gagné."""
        for line in self.WINNING_LINES:
            pieces = [self._board[pos] for pos in line]
            
            # Toutes les positions doivent être occupées
            if -1 in pieces:
                continue
            
            # Vérifier chaque attribut
            for bit in range(4):
                # Tous les 1 ou tous les 0 pour cet attribut
                attrs = [(p >> bit) & 1 for p in pieces]
                if all(a == 1 for a in attrs) or all(a == 0 for a in attrs):
                    return True
        
        return False
    
    def get_available_actions(self) -> List[int]:
        """Retourne les actions valides (0-15 place, 16-31 give)."""
        if self._done:
            return []
        
        if self._phase == "place":
            # Actions 0-15 : positions vides
            return [i for i in range(self.N_POSITIONS) if self._board[i] < 0]
        else:
            # Actions 16-31 : pièces disponibles (décalées de +16)
            return [p + self.N_POSITIONS for p in self._available_pieces]
    
    @property
    def is_game_over(self) -> bool:
        return self._done
    
    @property
    def current_player(self) -> int:
        return self._current_player
    
    def clone(self) -> "Quarto":
        """Clone l'environnement."""
        env = Quarto()
        env._board = self._board.copy()
        env._available_pieces = self._available_pieces.copy()
        env._current_piece = self._current_piece
        env._current_player = self._current_player
        env._phase = self._phase
        env._winner = self._winner
        env._done = self._done
        env._move_count = self._move_count
        return env
    
    def render(self) -> None:
        """Affiche le plateau."""
        print(f"\n{'='*40}")
        print(f"Quarto - Tour: Joueur {self._current_player + 1}")
        print(f"Phase: {self._phase.upper()}")
        if self._current_piece is not None:
            piece = QuartoPiece.from_id(self._current_piece)
            print(f"Pièce à placer: {piece} (ID: {self._current_piece})")
        print()
        
        # Afficher le plateau
        print("+------+------+------+------+")
        for row in range(4):
            line = "|"
            for col in range(4):
                pos = row * 4 + col
                piece_id = self._board[pos]
                if piece_id >= 0:
                    piece = QuartoPiece.from_id(piece_id)
                    line += f" {piece} |"
                else:
                    line += f"  {pos:2d}  |"
            print(line)
            print("+------+------+------+------+")
        
        # Pièces disponibles
        print(f"\nPièces disponibles: {sorted(self._available_pieces)}")
        
        if self._done:
            if self._winner >= 0:
                print(f"\nVictoire: Joueur {self._winner + 1}!")
            else:
                print("\nMatch nul!")
    
    def get_symmetries(
        self,
        state: np.ndarray,
        policy: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Retourne les 8 symétries (4 rotations × 2 réflexions) de l'état et de la politique.

        Le plateau 4×4 possède le même groupe de symétries D4 qu'un carré.
        Chaque transformation permute les 16 positions du plateau de manière
        cohérente dans le vecteur d'état **et** dans le vecteur de politique.

        Note : seule la phase "place" bénéficie de la permutation de politique.
        En phase "give", les actions désignent des pièces (pas des positions),
        donc la politique n'est pas permutée — mais l'état l'est quand même.
        """
        symmetries: List[Tuple[np.ndarray, np.ndarray]] = []

        # Les 8 transformations D4 exprimées comme permutations d'indices 0-15
        # Convention : indice = row * 4 + col  (row,col dans 0..3)
        def _idx(r: int, c: int) -> int:
            return r * 4 + c

        transforms: List[List[int]] = []
        for rot in range(4):          # 0°, 90°, 180°, 270°
            for flip in (False, True): # identité / réflexion horizontale
                perm = [0] * 16
                for r in range(4):
                    for c in range(4):
                        rr, cc = r, c
                        # Rotations successives de 90° (sens horaire)
                        for _ in range(rot):
                            rr, cc = cc, 3 - rr
                        # Réflexion horizontale (miroir gauche-droite)
                        if flip:
                            cc = 3 - cc
                        perm[_idx(r, c)] = _idx(rr, cc)
                transforms.append(perm)

        for perm in transforms:
            new_state = self._permute_board_in_state(state, perm)
            new_policy = np.zeros_like(policy)
            # Permuter la partie "place" (indices 0-15)
            for src, dst in enumerate(perm):
                new_policy[dst] = policy[src]
            # La partie "give" (indices 16-31) n'est pas permutée
            new_policy[self.N_POSITIONS:] = policy[self.N_POSITIONS:]
            symmetries.append((new_state, new_policy))

        return symmetries

    # ------------------------------------------------------------------
    def _permute_board_in_state(
        self, state: np.ndarray, perm: List[int]
    ) -> np.ndarray:
        """Applique une permutation de positions au vecteur d'état encodé."""
        new_state = state.copy()
        # Compact : 16 blocs de 5 (board) puis le reste inchangé
        for src, dst in enumerate(perm):
            new_state[dst * 5 : dst * 5 + 5] = state[src * 5 : src * 5 + 5]
        return new_state


class QuartoVsRandom(Quarto):
    """
    Quarto contre un adversaire aléatoire.
    
    L'agent joue toujours en tant que joueur 0.
    Le joueur 1 joue aléatoirement.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "QuartoVsRandom"
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Effectue l'action et fait jouer l'adversaire si nécessaire."""
        state, reward, done = super().step(action)
        
        # Si c'est au tour de l'adversaire et pas fini
        while not done and self._current_player == 1:
            # Adversaire joue aléatoirement
            available = self.get_available_actions()
            if len(available) == 0:
                break
            opponent_action = self.rng.choice(available)
            state, opp_reward, done = super().step(opponent_action)
            
            # Inverser la récompense pour l'agent
            if done and self._winner == 1:
                reward = -1.0
        
        return state, reward, done


# Test
if __name__ == "__main__":
    print("=== Test de Quarto ===\n")
    
    env = Quarto()
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State dim: {env.state_dim}")
    print(f"N actions: {env.n_actions}")
    
    env.render()
    
    # Jouer quelques coups
    print("\n--- Partie de test ---")
    
    # J1 donne pièce 0 (action 16 = pièce 0)
    state, reward, done = env.step(16)
    print("J1 donne pièce 0")
    
    # J2 place en position 0 (action 0)
    state, reward, done = env.step(0)
    print("J2 place en position 0")
    env.render()
    
    # J2 donne pièce 1 (action 17 = pièce 1)
    state, reward, done = env.step(17)
    print("J2 donne pièce 1")
    
    # J1 place en position 5 (action 5)
    state, reward, done = env.step(5)
    print("J1 place en position 5")
    env.render()
    
    print("\n--- Test QuartoVsRandom ---")
    env2 = QuartoVsRandom()
    state = env2.reset()
    
    step = 0
    while not env2.is_game_over and step < 50:
        available = env2.get_available_actions()
        if len(available) == 0:
            break
        action = np.random.choice(available)
        state, reward, done = env2.step(action)
        step += 1
    
    env2.render()
    print(f"\nPartie terminée en {step} coups")
    
    print("\n[OK] Tests passes!")
