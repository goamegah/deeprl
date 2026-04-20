"""
Interface abstraite pour les environnements.

Cette classe définit le contrat que tout environnement doit respecter.
Elle s'inspire de l'API Gymnasium (anciennement OpenAI Gym) mais reste simple.

Concepts clés:
- État (state): La représentation actuelle de l'environnement
- Action: Une décision que l'agent peut prendre
- Récompense (reward): Un signal numérique indiquant la qualité d'une action
- Terminaison: Indique si l'épisode est fini
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class Environment(ABC):
    """
    Classe abstraite définissant l'interface d'un environnement.
    
    Un environnement est un monde dans lequel un agent peut agir.
    L'agent observe l'état, choisit une action, et reçoit une récompense.
    
    Attributs:
        name (str): Nom de l'environnement
        state_shape (Tuple): Forme de l'espace d'états
        n_actions (int): Nombre d'actions possibles
        current_player (int): Joueur actuel (pour jeux multi-joueurs)
    """
    
    def __init__(self, name: str = "Environment"):
        """
        Initialise l'environnement.
        
        Args:
            name: Nom descriptif de l'environnement
        """
        self.name = name
        self._done: bool = False
        self._current_player: int = 0  # Pour les jeux à plusieurs joueurs
    
    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """
        Retourne la forme de l'espace d'états.
        
        Returns:
            Tuple décrivant les dimensions de l'état.
            Ex: (4,) pour un vecteur 1D de taille 4
                (8, 8) pour une grille 8x8
        """
        pass
    
    @property
    @abstractmethod
    def n_actions(self) -> int:
        """
        Retourne le nombre total d'actions possibles.
        
        Returns:
            Nombre d'actions dans l'espace d'actions discret.
        """
        pass
    
    @property
    def state_dim(self) -> int:
        """
        Retourne la dimension aplatie de l'état.
        
        Utile pour les réseaux de neurones qui prennent des vecteurs en entrée.
        
        Returns:
            Produit de toutes les dimensions de state_shape.
        """
        return int(np.prod(self.state_shape))
    
    @property
    def current_player(self) -> int:
        """
        Retourne l'identifiant du joueur courant.
        
        Pour les environnements mono-joueur, retourne toujours 0.
        Pour les jeux à 2 joueurs, alterne entre 0 et 1.
        
        Returns:
            Identifiant du joueur (0, 1, ...)
        """
        return self._current_player
    
    @property
    def is_game_over(self) -> bool:
        """
        Indique si l'épisode/partie est terminé.
        
        Returns:
            True si terminé, False sinon.
        """
        return self._done
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement à son état initial.
        
        Cette méthode doit être appelée au début de chaque épisode.
        
        Returns:
            L'état initial de l'environnement.
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Exécute une action dans l'environnement.
        
        C'est la méthode principale d'interaction avec l'environnement.
        
        Args:
            action: L'indice de l'action à exécuter (0 <= action < n_actions)
        
        Returns:
            Tuple (next_state, reward, done):
                - next_state: Le nouvel état après l'action
                - reward: La récompense obtenue
                - done: True si l'épisode est terminé
        """
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[int]:
        """
        Retourne la liste des actions valides dans l'état courant.
        
        Certains environnements ont des actions qui ne sont pas toujours valides
        (ex: placer une pièce sur une case déjà occupée au TicTacToe).
        
        Returns:
            Liste des indices d'actions valides.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Retourne l'état courant de l'environnement.
        
        Returns:
            L'état actuel sous forme de numpy array.
        """
        pass
    
    def action_mask(self) -> np.ndarray:
        """
        Masque booléen des actions légales (taille n_actions).
        
        Utilisé par les réseaux de neurones pour masquer les logits
        des actions illégales avant le softmax.
        
        Returns:
            Array booléen, True = action légale.
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        for a in self.get_available_actions():
            mask[a] = True
        return mask
    
    def render(self, mode: str = "text") -> Optional[str]:
        """
        Affiche l'environnement.
        
        Args:
            mode: "text" pour affichage console, "rgb" pour image
        
        Returns:
            Représentation textuelle si mode="text", None sinon.
        """
        # Implémentation par défaut
        return str(self.get_state())
    
    def clone(self) -> "Environment":
        """
        Crée une copie profonde de l'environnement.
        
        Utile pour les algorithmes de planification (MCTS, etc.)
        qui ont besoin de simuler des actions sans modifier l'état réel.
        
        Returns:
            Une copie indépendante de l'environnement.
        """
        import copy
        return copy.deepcopy(self)
    
    @abstractmethod
    def determinize(self, obs: np.ndarray) -> "Environment":
        """
        Reconstruit un environnement jouable à partir d'une observation.
        
        Crée un clone dont l'état interne est cohérent avec l'observation
        donnée. Indispensable pour les algorithmes de planification (MCTS)
        qui simulent à partir d'un état observé.
        
        - Jeux à information parfaite : décode l'observation en état complet.
        - Jeux à information imparfaite : échantillonne l'information cachée.
        - Jeux à 2 joueurs : restaure le joueur courant, la phase de jeu,
          et toute information nécessaire pour continuer la partie.
        
        Args:
            obs: Vecteur d'observation (tel que retourné par get_state).
        
        Returns:
            Un environnement indépendant, prêt à être joué.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}(state_shape={self.state_shape}, n_actions={self.n_actions})"
