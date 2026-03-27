"""
Interface abstraite pour les agents.

Un agent est une entité qui apprend à prendre des décisions dans un environnement.
Cette classe définit le contrat que tout agent doit respecter.

Cycle de vie d'un agent:
1. Création avec les paramètres de l'environnement
2. Appel de act() pour choisir une action
3. Appel de learn() pour mettre à jour l'agent (si applicable)
4. Répéter 2-3 jusqu'à la fin de l'épisode
5. Optionnel: save/load pour persister l'agent
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import torch


class Agent(ABC):
    """
    Classe abstraite définissant l'interface d'un agent.
    
    Attributs:
        name (str): Nom de l'agent
        state_dim (int): Dimension de l'espace d'états
        n_actions (int): Nombre d'actions possibles
        device (torch.device): Dispositif de calcul (CPU/GPU)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        name: str = "Agent",
        device: Optional[str] = None
    ):
        """
        Initialise l'agent.
        
        Args:
            state_dim: Dimension de l'espace d'états (taille du vecteur d'état)
            n_actions: Nombre d'actions possibles
            name: Nom descriptif de l'agent
            device: "cpu" ou "cuda" (auto-détection si None)
        """
        self.name = name
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Configuration du device (CPU ou GPU)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Statistiques d'entraînement
        self.training_steps = 0
        self.episodes_played = 0
    
    @abstractmethod
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action à partir de l'état courant.
        
        C'est la méthode principale de décision de l'agent.
        
        Args:
            state: L'état courant de l'environnement
            available_actions: Liste des actions valides (None = toutes)
            training: Si True, peut explorer. Si False, exploite uniquement.
            **kwargs: Arguments supplémentaires (ex: env pour MCTS/AlphaZero)
        
        Returns:
            L'indice de l'action choisie
        """
        pass
    
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
        Met à jour l'agent à partir d'une transition.
        
        Cette méthode est appelée après chaque step pour permettre
        à l'agent d'apprendre de son expérience.
        
        Args:
            state: État avant l'action
            action: Action effectuée
            reward: Récompense reçue
            next_state: État après l'action
            done: True si l'épisode est terminé
            **kwargs: Arguments supplémentaires (ex: available_actions_next)
        
        Returns:
            Dictionnaire de métriques d'apprentissage (loss, etc.) ou None
        """
        # Par défaut, pas d'apprentissage (ex: RandomAgent)
        return None
    
    def on_episode_start(self) -> None:
        """
        Appelée au début de chaque épisode.
        
        Permet de réinitialiser des états internes si nécessaire.
        """
        pass
    
    def on_episode_end(self, total_reward: float, episode_length: int) -> None:
        """
        Appelée à la fin de chaque épisode.
        
        Args:
            total_reward: Récompense totale de l'épisode
            episode_length: Nombre de steps dans l'épisode
        """
        self.episodes_played += 1
    
    def set_training_mode(self, training: bool) -> None:
        """
        Active ou désactive le mode entraînement.
        
        En mode entraînement, l'agent peut explorer.
        En mode évaluation, l'agent exploite uniquement.
        
        Args:
            training: True pour mode entraînement, False pour évaluation
        """
        # Par défaut, pas d'effet (à surcharger si nécessaire)
        pass
    
    def save(self, path: str) -> None:
        """
        Sauvegarde l'agent sur disque.
        
        Args:
            path: Chemin du fichier de sauvegarde
        """
        # Implémentation par défaut pour les agents simples
        save_dict = {
            "name": self.name,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
        }
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """
        Charge l'agent depuis le disque.
        
        Args:
            path: Chemin du fichier à charger
        """
        # Implémentation par défaut pour les agents simples
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.training_steps = checkpoint.get("training_steps", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de l'agent.
        
        Utile pour le logging et la reproductibilité.
        
        Returns:
            Dictionnaire de configuration
        """
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "device": str(self.device),
        }
    
    def __repr__(self) -> str:
        return f"{self.name}(state_dim={self.state_dim}, n_actions={self.n_actions})"
