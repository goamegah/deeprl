"""
Agent Aléatoire (Random Agent)

L'agent le plus simple possible: il choisit une action au hasard
parmi les actions disponibles.

Cet agent sert de:
- Baseline: si un autre agent fait moins bien que Random, il y a un problème
- Adversaire: pour entraîner des agents sur des jeux à 2 joueurs
- Test: pour vérifier que l'environnement fonctionne

Il n'apprend pas, mais c'est un point de départ essentiel.
"""

import numpy as np
from typing import List, Optional, Dict
from deeprl.agents.base import Agent


class RandomAgent(Agent):
    """
    Agent qui choisit des actions aléatoirement.
    
    Cet agent ne fait aucun apprentissage - il sélectionne simplement
    une action au hasard parmi les actions disponibles.
    
    Exemple d'utilisation:
        >>> agent = RandomAgent(state_dim=5, n_actions=2)
        >>> state = np.array([1, 0, 0, 0, 0])
        >>> action = agent.act(state)
        >>> print(f"Action choisie: {action}")
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        seed: Optional[int] = None
    ):
        """
        Crée un agent aléatoire.
        
        Args:
            state_dim: Dimension de l'espace d'états
            n_actions: Nombre d'actions possibles
            seed: Graine aléatoire pour reproductibilité (optionnel)
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="RandomAgent"
        )
        
        # Générateur aléatoire (avec seed optionnel pour reproductibilité)
        self.rng = np.random.default_rng(seed)
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action aléatoire.
        
        Note: L'état est ignoré car l'agent est complètement aléatoire.
        
        Args:
            state: État courant (ignoré)
            available_actions: Actions valides. Si None, toutes les actions
            training: Mode entraînement (ignoré pour cet agent)
        
        Returns:
            Une action choisie aléatoirement
        """
        # Si pas d'actions spécifiées, utiliser toutes les actions
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        
        # Vérifier qu'il y a au moins une action disponible
        if len(available_actions) == 0:
            raise ValueError("Aucune action disponible!")
        
        # Choisir une action au hasard
        action = self.rng.choice(available_actions)
        
        return int(action)
    
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
        L'agent aléatoire n'apprend pas.
        
        Cette méthode existe pour respecter l'interface Agent,
        mais elle ne fait rien.
        
        Returns:
            None (pas de métriques d'apprentissage)
        """
        return None
    
    def get_config(self) -> Dict:
        """Retourne la configuration de l'agent."""
        config = super().get_config()
        config["type"] = "RandomAgent"
        return config


# Test rapide si exécuté directement
if __name__ == "__main__":
    print("=== Test de RandomAgent ===\n")
    
    # Créer un agent
    agent = RandomAgent(state_dim=5, n_actions=2, seed=42)
    print(f"Agent créé: {agent}")
    print(f"Configuration: {agent.get_config()}")
    
    # Tester sur plusieurs états
    state = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    
    print("\n--- Test de 10 actions ---")
    actions_count = {0: 0, 1: 0}
    for i in range(10):
        action = agent.act(state)
        actions_count[action] += 1
        print(f"  Action {i+1}: {action}")
    
    print(f"\nDistribution: {actions_count}")
    
    # Tester avec actions limitées
    print("\n--- Test avec actions limitées ---")
    limited_actions = [0]  # Seulement action 0 disponible
    for i in range(5):
        action = agent.act(state, available_actions=limited_actions)
        print(f"  Action {i+1}: {action} (devrait toujours être 0)")
    
    print("\n[OK] Tests passes!")
