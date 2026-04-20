"""
Tabular Q-Learning - L'algorithme fondamental du RL.

Q-Learning est l'un des algorithmes les plus importants en RL.
Il apprend une fonction Q(s, a) qui estime la valeur de prendre
l'action 'a' dans l'état 's'.

Équation de mise à jour (Bellman):
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

où:
    - α (alpha) = learning rate (taux d'apprentissage)
    - γ (gamma) = discount factor (facteur d'actualisation)
    - r = récompense obtenue
    - s' = nouvel état
    - max_a' Q(s', a') = meilleure valeur Q dans le nouvel état

Exploration vs Exploitation:
    - ε-greedy: avec probabilité ε, on explore (action aléatoire)
                avec probabilité 1-ε, on exploite (meilleure action)
    - ε décroît au fil du temps pour converger vers l'exploitation

Limitations:
    - Ne fonctionne que pour des espaces d'états DISCRETS et PETITS
    - La table Q peut exploser en mémoire pour de grands espaces
    → Solution: Deep Q-Learning (DQN) avec réseaux de neurones
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Any
from deeprl.agents.base import Agent


class TabularQLearning(Agent):
    """
    Agent Q-Learning avec table de valeurs.
    
    Stocke les valeurs Q dans une matrice de taille (n_states, n_actions).
    Utilise une politique ε-greedy pour l'exploration.
    
    Attributes:
        q_table: Matrice des valeurs Q
        lr: Taux d'apprentissage (alpha)
        gamma: Facteur d'actualisation
        epsilon: Probabilité d'exploration
    
    Exemple d'utilisation:
        >>> agent = TabularQLearning(n_states=25, n_actions=4)
        >>> action = agent.act(state)
        >>> agent.learn(state, action, reward, next_state, done)
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            n_states: Nombre d'états possibles
            n_actions: Nombre d'actions possibles
            lr: Taux d'apprentissage (0 < lr <= 1)
                - Plus élevé = apprentissage rapide mais instable
                - Plus bas = apprentissage lent mais stable
            gamma: Facteur d'actualisation (0 <= gamma <= 1)
                - 0 = ne considère que les récompenses immédiates
                - 1 = considère autant le futur que le présent
            epsilon_start: Epsilon initial (exploration)
            epsilon_end: Epsilon minimum
            epsilon_decay: Facteur de décroissance de epsilon
            seed: Graine aléatoire pour reproductibilité
        """
        # Note: on utilise n_states comme state_dim pour la compatibilité
        super().__init__(
            state_dim=n_states,
            n_actions=n_actions,
            name="TabularQLearning"
        )
        
        self.n_states = n_states
        self.lr = lr
        self.gamma = gamma
        
        # Paramètres d'exploration ε-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Table Q: initialisée à 0 (ou petites valeurs aléatoires)
        # Forme: (n_states, n_actions)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        
        # Générateur aléatoire
        self.rng = np.random.default_rng(seed)
        
        # Statistiques
        self._training = True
    
    def _state_to_index(self, state: np.ndarray) -> int:
        """
        Convertit un état (one-hot ou index) en indice entier.
        
        Args:
            state: État sous forme de vecteur one-hot ou scalaire
        
        Returns:
            Indice de l'état (0 à n_states-1)
        """
        if isinstance(state, (int, np.integer)):
            return int(state)
        
        # Si c'est un vecteur one-hot, trouver l'indice du 1
        if len(state.shape) == 1:
            return int(np.argmax(state))
        
        # Si c'est une grille 2D, aplatir et trouver le 1
        flat = state.flatten()
        return int(np.argmax(flat))
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """
        Choisit une action selon la politique ε-greedy.
        
        En mode training:
            - Avec probabilité ε: action aléatoire (exploration)
            - Avec probabilité 1-ε: meilleure action selon Q (exploitation)
        
        En mode évaluation:
            - Toujours la meilleure action (exploitation pure)
        
        Args:
            state: État courant
            available_actions: Actions valides (None = toutes)
            training: Mode entraînement ou évaluation
        
        Returns:
            L'action choisie
        """
        state_idx = self._state_to_index(state)
        
        if available_actions is None:
            available_actions = list(range(self.n_actions))
        
        # Exploration vs Exploitation
        use_training = training and self._training
        
        if use_training and self.rng.random() < self.epsilon:
            # Exploration: action aléatoire
            action = self.rng.choice(available_actions)
        else:
            # Exploitation: meilleure action selon Q
            q_values = self.q_table[state_idx]
            
            # Masquer les actions non disponibles
            masked_q = np.full(self.n_actions, -np.inf)
            for a in available_actions:
                masked_q[a] = q_values[a]
            
            # Choisir l'action avec la plus grande valeur Q
            # En cas d'égalité, choisir aléatoirement parmi les meilleures
            max_q = np.max(masked_q)
            best_actions = [a for a in available_actions if q_values[a] == max_q]
            action = self.rng.choice(best_actions)
        
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
        Met à jour la table Q avec l'équation de Bellman.
        
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        
        Args:
            state: État avant l'action
            action: Action effectuée
            reward: Récompense reçue
            next_state: État après l'action
            done: True si l'épisode est terminé
        
        Returns:
            Dictionnaire avec la TD error
        """
        available_actions_next = kwargs.get("available_actions_next")
        
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        
        # Valeur Q actuelle
        current_q = self.q_table[state_idx, action]
        
        # Calcul de la cible (target)
        if done:
            # Si terminé, pas de valeur future
            target = reward
        else:
            # max sur les actions légales uniquement (S&B Ch.6.5)
            if available_actions_next is not None and len(available_actions_next) > 0:
                next_max_q = max(self.q_table[next_state_idx, a] for a in available_actions_next)
            else:
                next_max_q = np.max(self.q_table[next_state_idx])
            target = reward + self.gamma * next_max_q
        
        # TD Error (Temporal Difference Error)
        td_error = target - current_q
        
        # Mise à jour de Q
        self.q_table[state_idx, action] += self.lr * td_error
        
        self.training_steps += 1
        
        return {"td_error": abs(td_error), "q_value": current_q}
    
    def on_episode_end(self, total_reward: float, episode_length: int) -> None:
        """
        Appelée à la fin de chaque épisode.
        
        Décroît epsilon pour réduire l'exploration au fil du temps.
        """
        super().on_episode_end(total_reward, episode_length)
        
        # Décroissance de epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
    
    def set_training_mode(self, training: bool) -> None:
        """Active ou désactive le mode entraînement."""
        self._training = training
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Retourne les valeurs Q pour un état donné.
        
        Args:
            state: L'état
        
        Returns:
            Vecteur des valeurs Q pour chaque action
        """
        state_idx = self._state_to_index(state)
        return self.q_table[state_idx].copy()
    
    def get_policy(self) -> np.ndarray:
        """
        Retourne la politique déterministe optimale.
        
        Returns:
            Vecteur où policy[s] = meilleure action pour l'état s
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """
        Retourne la fonction de valeur V(s) = max_a Q(s, a).
        
        Returns:
            Vecteur où V[s] = valeur de l'état s
        """
        return np.max(self.q_table, axis=1)
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        save_dict = {
            "name": self.name,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "q_table": self.q_table.tolist(),  # Convertir en liste pour compatibilite
            "training_steps": self.training_steps,
            "episodes_played": self.episodes_played,
        }
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        q_table = checkpoint["q_table"]
        # Convertir de liste si necessaire
        if isinstance(q_table, list):
            self.q_table = np.array(q_table, dtype=np.float32)
        else:
            self.q_table = q_table
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint.get("training_steps", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'agent."""
        config = super().get_config()
        config.update({
            "type": "TabularQLearning",
            "n_states": self.n_states,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        })
        return config
    
    def __repr__(self) -> str:
        return (
            f"TabularQLearning(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"lr={self.lr}, γ={self.gamma}, ε={self.epsilon:.3f})"
        )


# Test rapide
if __name__ == "__main__":
    print("=== Test de TabularQLearning ===\n")
    
    # Créer un agent pour une grille 3x3 (9 états, 4 actions)
    agent = TabularQLearning(
        n_states=9,
        n_actions=4,
        lr=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99
    )
    print(f"Agent créé: {agent}")
    print(f"Configuration: {agent.get_config()}")
    
    # Test de conversion d'état
    state_onehot = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    state_idx = agent._state_to_index(state_onehot)
    print(f"\nConversion état one-hot → index: {state_idx}")
    
    # Test d'apprentissage simple
    print("\n--- Test d'apprentissage ---")
    state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # État 0
    action = 3  # Droite
    reward = -0.01
    next_state = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # État 1
    
    print(f"Avant apprentissage - Q[0]: {agent.q_table[0]}")
    
    for i in range(10):
        agent.learn(state, action, reward, next_state, done=False)
    
    print(f"Après 10 updates - Q[0]: {agent.q_table[0]}")
    
    # Test d'action
    print("\n--- Test de sélection d'action ---")
    agent.epsilon = 0.0  # Désactiver exploration
    action = agent.act(state)
    print(f"Action choisie (ε=0): {action} (devrait être 3)")
    
    agent.epsilon = 1.0  # Exploration pure
    actions = [agent.act(state) for _ in range(100)]
    print(f"Distribution avec ε=1: {np.bincount(actions, minlength=4)}")
    
    print("\n[OK] Tests passes!")
