"""
Monte Carlo Tree Search (MCTS) - Algorithme de planification.

MCTS est un algorithme de recherche qui combine:
- Simulation Monte Carlo (random rollouts)
- Recherche arborescente (tree search)
- UCB (Upper Confidence Bound) pour l'exploration

C'est l'algorithme derrière AlphaGo et AlphaZero.

Les 4 phases de MCTS:
1. Selection: descendre l'arbre en suivant UCB
2. Expansion: ajouter un nouveau nœud
3. Simulation (Rollout): jouer aléatoirement jusqu'à la fin
4. Backpropagation: remonter le résultat dans l'arbre

Formule UCB1:
    UCB(s, a) = Q(s, a) + c * √(ln(N(s)) / N(s, a))

où:
- Q(s, a) = valeur moyenne de l'action
- N(s) = nombre de visites de l'état
- N(s, a) = nombre de fois que l'action a été prise
- c = constante d'exploration (typiquement √2)

Référence:
- "A Survey of MCTS Methods" (Browne et al., 2012)
"""

import numpy as np
import math
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from deeprl.agents.base import Agent
from deeprl.envs.base import Environment


@dataclass
class MCTSNode:
    """
    Nœud de l'arbre MCTS.
    
    Stocke les statistiques de visites et les valeurs.
    """
    state: np.ndarray
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None  # Action qui a mené à ce nœud
    
    # Statistiques
    visits: int = 0
    value_sum: float = 0.0
    
    # Enfants
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    untried_actions: List[int] = field(default_factory=list)
    
    # Infos additionnelles
    is_terminal: bool = False
    player: int = 0  # Pour les jeux à 2 joueurs
    
    @property
    def value(self) -> float:
        """Valeur moyenne du nœud."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def is_fully_expanded(self) -> bool:
        """Tous les enfants ont été visités au moins une fois."""
        return len(self.untried_actions) == 0
    
    def ucb_score(self, c: float = math.sqrt(2), parent_player: int = 0) -> float:
        """
        Calcule le score UCB (Upper Confidence Bound).
        
        Chaque noeud stocke la valeur du point de vue de SON propre joueur.
        Lors de la sélection, le parent veut maximiser depuis SA perspective,
        donc on inverse l'exploitation quand les joueurs diffèrent.
        
        Args:
            c: Constante d'exploration
            parent_player: Joueur du noeud parent (pour ajuster la perspective)
        
        Returns:
            Score UCB
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value
        # Inverser si le parent est un joueur différent (sa victoire = notre défaite)
        if self.player != parent_player:
            exploitation = -exploitation
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def best_child(self, c: float = math.sqrt(2)) -> "MCTSNode":
        """Retourne l'enfant avec le meilleur score UCB."""
        return max(self.children.values(), key=lambda n: n.ucb_score(c, self.player))
    
    def best_action(self) -> int:
        """Retourne l'action la plus visitée (meilleure politique)."""
        return max(self.children.keys(), key=lambda a: self.children[a].visits)


class MCTSAgent(Agent):
    """
    Agent MCTS (Monte Carlo Tree Search).
    
    Utilise la simulation pour planifier les meilleures actions.
    Nécessite un environnement qui peut être cloné pour la simulation.
    
    Caractéristiques:
    - Pas d'apprentissage (planning pur)
    - Peut gérer des jeux à 2 joueurs
    - Qualité proportionnelle au temps de calcul
    
    Exemple d'utilisation:
        >>> agent = MCTSAgent(n_simulations=1000)
        >>> action = agent.act(state, env=env)
    """
    
    def __init__(
        self,
        state_dim: int = 0,  # Non utilisé, mais requis par l'interface
        n_actions: int = 0,  # Sera déterminé par l'environnement
        n_simulations: int = 100,
        c_exploration: float = math.sqrt(2),
        max_depth: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent MCTS.
        
        Args:
            state_dim: (Non utilisé directement)
            n_actions: (Déterminé par l'environnement)
            n_simulations: Nombre de simulations par décision
            c_exploration: Constante d'exploration UCB
            max_depth: Profondeur maximale des rollouts
            seed: Graine aléatoire
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="MCTS (UCT)"
        )
        
        self.n_simulations = n_simulations
        self.c = c_exploration
        self.max_depth = max_depth
        
        self.rng = np.random.default_rng(seed)
        
        # Statistiques
        self.total_simulations = 0
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        env: Optional[Environment] = None,
        **kwargs
    ) -> int:
        """
        Choisit une action en utilisant MCTS.
        
        IMPORTANT: Nécessite l'environnement pour simuler.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: (Non utilisé pour MCTS)
            env: Environnement clonable pour simulation
        
        Returns:
            Meilleure action selon MCTS
        """
        if env is None:
            raise ValueError("MCTS nécessite un environnement (env=...)")
        
        # Obtenir les actions disponibles
        if available_actions is None:
            available_actions = env.get_available_actions()
        
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Créer le nœud racine
        root = MCTSNode(
            state=state.copy(),
            untried_actions=available_actions.copy(),
            player=env.current_player
        )
        
        # Exécuter les simulations
        for _ in range(self.n_simulations):
            self._simulate(root, env.clone())
        
        self.total_simulations += self.n_simulations
        
        # Retourner l'action la plus visitée
        return root.best_action()
    
    def _simulate(self, root: MCTSNode, env: Environment) -> float:
        """
        Exécute une simulation MCTS complète.
        
        Args:
            root: Nœud racine
            env: Clone de l'environnement
        
        Returns:
            Valeur de la simulation
        """
        node = root
        
        # 1. SELECTION: descendre l'arbre
        while node.is_fully_expanded and len(node.children) > 0:
            node = node.best_child(self.c)
            
            # Appliquer l'action dans l'environnement
            if node.action is not None:
                _, _, done = env.step(node.action)
                if done:
                    break
        
        # 2. EXPANSION: ajouter un nouveau nœud
        if not node.is_terminal and len(node.untried_actions) > 0:
            action = self.rng.choice(node.untried_actions)
            node.untried_actions.remove(action)
            
            next_state, reward, done = env.step(action)
            
            child = MCTSNode(
                state=next_state.copy(),
                parent=node,
                action=action,
                is_terminal=done,
                untried_actions=env.get_available_actions() if not done else [],
                player=env.current_player
            )
            node.children[action] = child
            node = child
            
            if done:
                # La partie est terminée, utiliser la récompense directement
                value = reward
                self._backpropagate(node, value)
                return value
        
        # 3. SIMULATION (ROLLOUT): jouer aléatoirement
        value = self._rollout(env)
        
        # 4. BACKPROPAGATION: remonter la valeur
        self._backpropagate(node, value)
        
        return value
    
    def _rollout(self, env: Environment, perspective_player: Optional[int] = None) -> float:
        """
        Effectue un rollout aléatoire jusqu'à la fin.
        
        Args:
            env: Environnement (sera modifié)
            perspective_player: Player whose perspective the value is from.
                               Defaults to env.current_player (the player to
                               move at the start of the rollout).
        
        Returns:
            For 2-player zero-sum games: +1 (perspective_player wins),
                -1 (opponent wins), 0 (draw or incomplete).
            For 1-player games: accumulated reward.
        """
        if perspective_player is None:
            perspective_player = env.current_player
        
        total_reward = 0.0
        depth = 0
        
        while not env.is_game_over and depth < self.max_depth:
            actions = env.get_available_actions()
            if len(actions) == 0:
                break
            
            action = self.rng.choice(actions)
            _, reward, done = env.step(action)
            total_reward += reward
            depth += 1
        
        # For 2-player zero-sum games: return from perspective_player's view.
        # The raw total_reward is unreliable because step() returns +1 for
        # whichever player just won — summing these doesn't tell us WHO won.
        if hasattr(env, '_winner'):
            winner = env._winner
            if winner is not None and winner >= 0:
                return 1.0 if winner == perspective_player else -1.0
            return 0.0  # Draw or game not finished
        
        # For 1-player games: accumulated reward is correct
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Remonte la valeur dans l'arbre.
        
        Pour les jeux à 2 joueurs, la valeur est négée à chaque
        changement de joueur. Pour les jeux 1 joueur (player identique
        à chaque noeud), la valeur est propagée telle quelle.
        
        Args:
            node: Nœud de départ
            value: Valeur à propager
        """
        while node is not None:
            node.visits += 1
            node.value_sum += value
            parent = node.parent
            
            # Néger seulement si le joueur change entre parent et enfant
            if parent is not None and node.player != parent.player:
                value = -value
            
            node = parent
    
    def learn(self, *args, **kwargs) -> None:
        """MCTS n'apprend pas (planning pur)."""
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            "type": "MCTS",
            "n_simulations": self.n_simulations,
            "c_exploration": self.c,
            "max_depth": self.max_depth,
        })
        return config


class RandomRolloutAgent(Agent):
    """
    Agent Random Rollout - Version simplifiée de MCTS.
    
    Pour chaque action disponible, effectue plusieurs rollouts
    aléatoires et choisit l'action avec la meilleure moyenne.
    
    Plus simple que MCTS complet mais moins efficace car
    pas de construction d'arbre ni UCB.
    """
    
    def __init__(
        self,
        state_dim: int = 0,
        n_actions: int = 0,
        n_rollouts: int = 10,
        max_depth: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialise l'agent Random Rollout.
        
        Args:
            state_dim: (Non utilisé)
            n_actions: (Déterminé par l'environnement)
            n_rollouts: Nombre de rollouts par action
            max_depth: Profondeur maximale
            seed: Graine aléatoire
        """
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name="RandomRollout"
        )
        
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        env: Optional[Environment] = None,
        **kwargs
    ) -> int:
        """
        Choisit une action par random rollout.
        
        Args:
            state: État courant
            available_actions: Actions valides
            training: (Non utilisé)
            env: Environnement clonable
        
        Returns:
            Meilleure action selon les rollouts
        """
        if env is None:
            raise ValueError("RandomRollout nécessite un environnement")
        
        if available_actions is None:
            available_actions = env.get_available_actions()
        
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Évaluer chaque action
        action_values = {}
        
        for action in available_actions:
            values = []
            
            for _ in range(self.n_rollouts):
                # Cloner l'environnement
                env_clone = env.clone()
                
                # The calling player (who chose this action)
                calling_player = env_clone.current_player
                
                # Appliquer l'action
                _, reward, done = env_clone.step(action)
                total_reward = reward
                
                # Rollout from the calling player's perspective
                if not done:
                    total_reward += self._rollout(
                        env_clone, perspective_player=calling_player
                    )
                
                values.append(total_reward)
            
            action_values[action] = np.mean(values)
        
        # Retourner la meilleure action
        return max(action_values.keys(), key=lambda a: action_values[a])
    
    def _rollout(self, env: Environment, perspective_player: Optional[int] = None) -> float:
        """
        Rollout aléatoire.
        
        Args:
            env: Environnement (sera modifié)
            perspective_player: Player whose perspective the value is from.
                               Defaults to env.current_player.
        
        Returns:
            For 2-player games: +1/-1/0 from perspective_player's view.
            For 1-player games: accumulated reward.
        """
        if perspective_player is None:
            perspective_player = env.current_player
        
        total_reward = 0.0
        depth = 0
        
        while not env.is_game_over and depth < self.max_depth:
            actions = env.get_available_actions()
            if len(actions) == 0:
                break
            
            action = self.rng.choice(actions)
            _, reward, done = env.step(action)
            total_reward += reward
            depth += 1
        
        # For 2-player zero-sum games
        if hasattr(env, '_winner'):
            winner = env._winner
            if winner is not None and winner >= 0:
                return 1.0 if winner == perspective_player else -1.0
            return 0.0
        
        return total_reward
    
    def learn(self, *args, **kwargs) -> None:
        """Random Rollout n'apprend pas."""
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'agent."""
        config = super().get_config()
        config.update({
            "type": "RandomRollout",
            "n_rollouts": self.n_rollouts,
            "max_depth": self.max_depth,
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test de MCTS ===\n")
    
    # Importer TicTacToe pour le test
    from deeprl.envs.tictactoe import TicTacToe
    
    # Créer l'environnement et les agents
    env = TicTacToe()
    
    print("1. Test RandomRollout:")
    rollout_agent = RandomRolloutAgent(n_rollouts=10)
    print(f"   {rollout_agent.name}")
    
    env.reset()
    action = rollout_agent.act(env.get_state(), env=env)
    print(f"   Action choisie: {action}")
    
    print("\n2. Test MCTS:")
    mcts_agent = MCTSAgent(n_simulations=100)
    print(f"   {mcts_agent.name}")
    
    env.reset()
    action = mcts_agent.act(env.get_state(), env=env)
    print(f"   Action choisie: {action}")
    print(f"   Total simulations: {mcts_agent.total_simulations}")
    
    # Partie complète
    print("\n3. Partie MCTS vs Random:")
    env = TicTacToe()
    state = env.reset()
    
    mcts = MCTSAgent(n_simulations=50)
    
    while not env.is_game_over:
        available = env.get_available_actions()
        
        if env.current_player == 0:
            # MCTS joue
            action = mcts.act(state, available_actions=available, env=env)
            player = "MCTS"
        else:
            # Random joue
            action = np.random.choice(available)
            player = "Random"
        
        state, reward, done = env.step(action)
        print(f"   {player} joue {action}")
    
    env.render()
    
    print("\n[OK] Tests passes!")
