"""
Human Agent - Agent contrôlé par un joueur humain.

Permet à un humain de jouer dans les environnements via:
- La console (input texte)
- L'interface graphique Pygame (à utiliser avec GameViewer)

Cet agent satisfait l'exigence du projet:
"mettre à disposition un agent 'humain'"
"""

import numpy as np
from typing import List, Optional, Dict, Any

from deeprl.agents.base import Agent
from deeprl.envs.base import Environment


class HumanAgent(Agent):
    """
    Agent contrôlé par un humain.
    
    Modes d'interaction:
    - Console: l'humain entre l'action au clavier
    - GUI: l'action est fournie par l'interface graphique
    
    Exemple d'utilisation (console):
        >>> agent = HumanAgent(state_dim=9, n_actions=9)
        >>> env = TicTacToe()
        >>> state = env.reset()
        >>> action = agent.act(state, env.get_available_actions(), env=env)
    
    Exemple d'utilisation (GUI):
        >>> viewer = GameViewer(env, agent=None)  # agent=None active mode humain
        >>> viewer.run()
    """
    
    def __init__(
        self,
        state_dim: int = 0,
        n_actions: int = 0,
        mode: str = "console",  # "console" ou "gui"
        name: str = "Human"
    ):
        """
        Initialise l'agent humain.
        
        Args:
            state_dim: Dimension de l'état (pour compatibilité)
            n_actions: Nombre d'actions possibles
            mode: "console" pour input texte, "gui" pour Pygame
            name: Nom de l'agent
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name=name
        )
        
        self.mode = mode
        self._pending_action: Optional[int] = None
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = False,
        env: Optional[Environment] = None,
        **kwargs
    ) -> int:
        """
        Demande une action à l'humain.
        
        En mode console: affiche l'état et attend l'input
        En mode GUI: retourne l'action en attente (set_action)
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: Ignoré (humain ne s'entraîne pas)
            env: Environnement (pour affichage)
        
        Returns:
            Action choisie par l'humain
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        
        if self.mode == "gui":
            return self._act_gui(available_actions)
        else:
            return self._act_console(state, available_actions, env)
    
    def _act_console(
        self,
        state: np.ndarray,
        available_actions: List[int],
        env: Optional[Environment]
    ) -> int:
        """Mode console: affiche et demande l'input."""
        
        # Afficher l'état si environnement disponible
        if env is not None:
            print("\n" + "="*40)
            env.render()
        else:
            print(f"\nÉtat: {state}")
        
        print(f"\nActions disponibles: {available_actions}")
        
        while True:
            try:
                user_input = input("Votre action: ").strip()
                
                # Permettre 'q' pour quitter
                if user_input.lower() == 'q':
                    raise KeyboardInterrupt("Joueur a quitté")
                
                action = int(user_input)
                
                if action in available_actions:
                    return action
                else:
                    print(f"/!\\ Action {action} invalide. Choisissez parmi {available_actions}")
                    
            except ValueError:
                print("/!\\ Entrez un nombre valide.")
    
    def _act_gui(self, available_actions: List[int]) -> int:
        """
        Mode GUI: retourne l'action définie par set_action.
        
        L'interface graphique doit appeler set_action() avant act().
        """
        if self._pending_action is None:
            # Pas d'action en attente - attendre
            raise RuntimeError(
                "HumanAgent en mode GUI: appelez set_action() avant act()"
            )
        
        action = self._pending_action
        self._pending_action = None
        
        if action not in available_actions:
            raise ValueError(f"Action {action} non disponible")
        
        return action
    
    def set_action(self, action: int) -> None:
        """
        Définit l'action pour le mode GUI.
        
        Appelé par l'interface graphique quand l'humain clique.
        
        Args:
            action: Action choisie par l'humain
        """
        self._pending_action = action
    
    def has_pending_action(self) -> bool:
        """Vérifie si une action est en attente (mode GUI)."""
        return self._pending_action is not None
    
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
        L'agent humain n'apprend pas automatiquement.
        
        Retourne None.
        """
        return None
    
    def set_training_mode(self, training: bool) -> None:
        """L'agent humain n'a pas de mode training."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            "type": "Human",
            "mode": self.mode,
        })
        return config
    
    def save(self, path: str) -> None:
        """L'agent humain n'a rien à sauvegarder."""
        pass
    
    def load(self, path: str) -> None:
        """L'agent humain n'a rien à charger."""
        pass


# Test
if __name__ == "__main__":
    print("=== Test HumanAgent ===\n")
    
    # Test mode console
    print("1. Test mode console:")
    agent = HumanAgent(n_actions=4, mode="console")
    print(f"   Agent: {agent}")
    
    # Simuler (ne pas exécuter en mode automatique)
    print("   (Pour tester: python -c \"from deeprl.agents.human_agent import HumanAgent; ...\")")
    
    # Test mode GUI
    print("\n2. Test mode GUI:")
    agent = HumanAgent(n_actions=9, mode="gui")
    
    # Simuler l'interface graphique
    agent.set_action(4)
    assert agent.has_pending_action()
    action = agent._act_gui([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert action == 4
    assert not agent.has_pending_action()
    print("   ✓ set_action et act fonctionnent")
    
    print("\n[OK] Tests passés!")
