"""
Expert Apprentice - Apprentissage par imitation.

L'apprentissage par imitation consiste à apprendre une politique
à partir de démonstrations d'un expert (humain ou algorithme).

Deux approches principales:

1. Behavior Cloning (BC):
   - Apprentissage supervisé direct sur les couples (state, action_expert)
   - Simple mais souffre de distribution shift
   
2. DAgger (Dataset Aggregation):
   - Itère entre exécution de l'apprenti et re-labeling par l'expert
   - Corrige le distribution shift
   - Plus robuste mais nécessite un expert interactif

L'expert peut être:
- Un humain (via interface)
- MCTS avec beaucoup de simulations
- Un agent déjà entraîné
- Une politique optimale connue

Référence:
- "A Reduction of Imitation Learning to No-Regret Online Learning" (Ross et al., 2011)
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from deeprl.agents.base import Agent
from deeprl.envs.base import Environment
from deeprl.networks.mlp import MLP


class ExpertPolicy(ABC):
    """
    Interface pour une politique experte.
    
    L'expert fournit l'action optimale (ou une distribution)
    pour un état donné.
    """
    
    @abstractmethod
    def get_action(
        self,
        state: np.ndarray,
        env: Optional[Environment] = None,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        Retourne l'action de l'expert.
        
        Args:
            state: État courant
            env: Environnement (pour les experts basés sur simulation)
            available_actions: Actions valides
        
        Returns:
            Action recommandée par l'expert
        """
        pass
    
    def get_action_probs(
        self,
        state: np.ndarray,
        env: Optional[Environment] = None,
        available_actions: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Retourne la distribution sur les actions.
        
        Par défaut, one-hot sur l'action choisie.
        """
        action = self.get_action(state, env, available_actions)
        n_actions = len(available_actions) if available_actions else 10
        probs = np.zeros(n_actions, dtype=np.float32)
        probs[action] = 1.0
        return probs


class MCTSExpert(ExpertPolicy):
    """
    Expert basé sur MCTS.
    
    Utilise beaucoup de simulations pour trouver
    la meilleure action.
    """
    
    def __init__(
        self,
        n_simulations: int = 500,
        c_exploration: float = 1.41,
        seed: Optional[int] = None
    ):
        from deeprl.agents.planning.mcts import MCTSAgent
        
        self.name = f"MCTSExpert({n_simulations} sims)"
        self.mcts = MCTSAgent(
            n_simulations=n_simulations,
            c_exploration=c_exploration,
            seed=seed
        )
    
    def get_action(
        self,
        state: np.ndarray,
        env: Optional[Environment] = None,
        available_actions: Optional[List[int]] = None
    ) -> int:
        if env is None:
            raise ValueError("MCTSExpert nécessite l'environnement")
        
        return self.mcts.act(state, available_actions, env=env)


class HumanExpert(ExpertPolicy):
    """
    Expert humain (via console).
    """
    
    def __init__(self):
        self.name = "HumanExpert"
    
    def get_action(
        self,
        state: np.ndarray,
        env: Optional[Environment] = None,
        available_actions: Optional[List[int]] = None
    ) -> int:
        if env is not None:
            env.render()
        
        print(f"Actions disponibles: {available_actions}")
        
        while True:
            try:
                action = int(input("Votre action: "))
                if available_actions is None or action in available_actions:
                    return action
                print(f"Action invalide. Choisissez parmi {available_actions}")
            except ValueError:
                print("Entrez un nombre valide.")


class BehaviorCloning:
    """
    Behavior Cloning - Apprentissage supervisé simple.
    
    Apprend une politique par imitation directe des
    couples (état, action) de l'expert.
    
    Avantages:
    - Simple à implémenter
    - Ne nécessite pas d'environnement
    - Rapide à entraîner
    
    Inconvénients:
    - Distribution shift: les erreurs s'accumulent
    - Nécessite beaucoup de données
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Réseau de politique
        self.policy_net = MLP(
            state_dim=state_dim,
            output_dim=n_actions,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Données
        self.demonstrations: List[Tuple[np.ndarray, int]] = []
    
    def add_demonstration(self, state: np.ndarray, action: int):
        """Ajoute une démonstration de l'expert."""
        self.demonstrations.append((state.copy(), action))
    
    def add_demonstrations(
        self,
        states: List[np.ndarray],
        actions: List[int]
    ):
        """Ajoute plusieurs démonstrations."""
        for s, a in zip(states, actions):
            self.demonstrations.append((s.copy(), a))
    
    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Entraîne le modèle sur les démonstrations.
        
        Returns:
            Métriques d'entraînement
        """
        if len(self.demonstrations) == 0:
            raise ValueError("Pas de démonstrations!")
        
        self.policy_net.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        for epoch in range(n_epochs):
            # Mélanger les données
            indices = np.random.permutation(len(self.demonstrations))
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Utiliser np.array pour éviter le warning de conversion lente
                states = torch.FloatTensor(np.array([
                    self.demonstrations[j][0] for j in batch_indices
                ])).to(self.device)
                
                actions = torch.LongTensor(np.array([
                    self.demonstrations[j][1] for j in batch_indices
                ])).to(self.device)
                
                # Forward
                logits = self.policy_net(states)
                loss = F.cross_entropy(logits, actions)
                
                # Accuracy
                preds = logits.argmax(dim=-1)
                acc = (preds == actions).float().mean()
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_batches += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}: "
                      f"loss={epoch_loss/epoch_batches:.4f}, "
                      f"acc={epoch_acc/epoch_batches*100:.1f}%")
            
            total_loss += epoch_loss
            total_acc += epoch_acc
            n_batches += epoch_batches
        
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
        }
    
    def predict(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """Prédit l'action pour un état."""
        self.policy_net.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_net(state_tensor)
            
            if available_actions is not None:
                mask = torch.full((self.n_actions,), float('-inf')).to(self.device)
                for a in available_actions:
                    mask[a] = 0
                logits = logits + mask
            
            action = logits.argmax(dim=-1)
        
        return int(action.item())


class DAgger:
    """
    DAgger - Dataset Aggregation.
    
    Corrige le distribution shift en:
    1. Exécutant la politique actuelle
    2. Demandant à l'expert de labeler les états visités
    3. Ajoutant ces données et ré-entraînant
    
    Plus robuste que Behavior Cloning simple.
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        expert: ExpertPolicy,
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        beta_start: float = 1.0,  # Probabilité d'utiliser l'expert
        beta_decay: float = 0.9,
        device: Optional[str] = None
    ):
        self.expert = expert
        self.beta = beta_start
        self.beta_decay = beta_decay
        
        # Behavior Cloning sous-jacent
        self.bc = BehaviorCloning(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dims=hidden_dims,
            lr=lr,
            device=device
        )
    
    def collect_and_train(
        self,
        env: Environment,
        n_episodes: int = 10,
        n_epochs_train: int = 50,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Collecte des données et entraîne le modèle.
        
        Args:
            env: Environnement
            n_episodes: Épisodes de collecte
            n_epochs_train: Epochs d'entraînement par itération
            verbose: Afficher la progression
        
        Returns:
            Métriques
        """
        new_demos = 0
        
        for ep in range(n_episodes):
            state = env.reset()
            
            while not env.is_game_over:
                available = env.get_available_actions()
                
                # Obtenir l'action de l'expert
                expert_action = self.expert.get_action(state, env, available)
                
                # Décider qui joue
                if np.random.random() < self.beta:
                    action = expert_action
                else:
                    action = self.bc.predict(state, available)
                
                # Ajouter la démonstration
                self.bc.add_demonstration(state, expert_action)
                new_demos += 1
                
                # Exécuter l'action
                state, _, done = env.step(action)
        
        # Decay beta
        self.beta *= self.beta_decay
        
        # Entraîner
        metrics = self.bc.train(
            n_epochs=n_epochs_train,
            batch_size=32,
            verbose=verbose
        )
        
        metrics['new_demonstrations'] = new_demos
        metrics['total_demonstrations'] = len(self.bc.demonstrations)
        metrics['beta'] = self.beta
        
        return metrics
    
    def predict(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """Prédit l'action."""
        return self.bc.predict(state, available_actions)


class ExpertApprenticeAgent(Agent):
    """
    Agent Expert-Apprentice complet.
    
    Combine Behavior Cloning et DAgger pour apprendre
    d'un expert de manière robuste.
    
    Modes:
    - "bc": Behavior Cloning simple
    - "dagger": DAgger (plus robuste)
    
    Exemple:
        >>> expert = MCTSExpert(n_simulations=500)
        >>> agent = ExpertApprenticeAgent(
        ...     state_dim=27, n_actions=9,
        ...     expert=expert, mode="dagger"
        ... )
        >>> agent.train(env, n_iterations=10)
        >>> action = agent.act(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        expert: ExpertPolicy,
        mode: str = "dagger",
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            name=f"ExpertApprentice ({mode})",
            device=device
        )
        
        self.expert = expert
        self.mode = mode
        self.hidden_dims = hidden_dims
        
        if mode == "bc":
            self.learner = BehaviorCloning(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dims=hidden_dims,
                lr=lr,
                device=device
            )
        elif mode == "dagger":
            self.learner = DAgger(
                state_dim=state_dim,
                n_actions=n_actions,
                expert=expert,
                hidden_dims=hidden_dims,
                lr=lr,
                device=device
            )
        else:
            raise ValueError(f"Mode inconnu: {mode}")
        
        if seed is not None:
            np.random.seed(seed)
    
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = False,
        **kwargs
    ) -> int:
        """Choisit une action."""
        return self.learner.predict(state, available_actions)
    
    def collect_demonstrations(
        self,
        env: Environment,
        n_episodes: int = 100,
        verbose: bool = False
    ):
        """
        Collecte des démonstrations de l'expert.
        
        Pour Behavior Cloning uniquement.
        """
        if self.mode != "bc":
            raise ValueError("Utilisez train() pour DAgger")
        
        for ep in range(n_episodes):
            state = env.reset()
            
            while not env.is_game_over:
                available = env.get_available_actions()
                action = self.expert.get_action(state, env, available)
                
                self.learner.add_demonstration(state, action)
                
                state, _, done = env.step(action)
            
            if verbose and (ep + 1) % 10 == 0:
                print(f"Collecte: {ep + 1}/{n_episodes} épisodes, "
                      f"{len(self.learner.demonstrations)} démonstrations")
    
    def train(
        self,
        env: Environment,
        n_iterations: int = 10,
        episodes_per_iteration: int = 20,
        epochs_per_iteration: int = 50,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Entraîne l'agent par imitation.
        
        Args:
            env: Environnement
            n_iterations: Nombre d'itérations de collecte+entraînement
            episodes_per_iteration: Épisodes par itération
            epochs_per_iteration: Epochs d'entraînement par itération
            verbose: Afficher la progression
        
        Returns:
            Historique des métriques
        """
        history = []
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Itération {iteration + 1}/{n_iterations} ---")
            
            if self.mode == "bc":
                # Collecte puis entraînement
                self.collect_demonstrations(
                    env, n_episodes=episodes_per_iteration, verbose=False
                )
                
                metrics = self.learner.train(
                    n_epochs=epochs_per_iteration,
                    verbose=verbose
                )
                metrics['demonstrations'] = len(self.learner.demonstrations)
            else:
                # DAgger: collecte et entraînement combinés
                metrics = self.learner.collect_and_train(
                    env,
                    n_episodes=episodes_per_iteration,
                    n_epochs_train=epochs_per_iteration,
                    verbose=verbose
                )
            
            history.append(metrics)
            
            if verbose:
                print(f"Loss: {metrics['loss']:.4f}, "
                      f"Acc: {metrics['accuracy']*100:.1f}%")
            
            self.training_steps += 1
        
        return history
    
    def learn(self, *args, **kwargs) -> None:
        """Non utilisé pour l'imitation."""
        return None
    
    def save(self, path: str) -> None:
        """Sauvegarde l'agent."""
        if self.mode == "bc":
            policy_net = self.learner.policy_net
        else:
            policy_net = self.learner.bc.policy_net
        
        torch.save({
            'policy_net': policy_net.state_dict(),
            'training_steps': self.training_steps,
            'config': self.get_config(),
        }, path)
    
    def load(self, path: str) -> None:
        """Charge l'agent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if self.mode == "bc":
            self.learner.policy_net.load_state_dict(checkpoint['policy_net'])
        else:
            self.learner.bc.policy_net.load_state_dict(checkpoint['policy_net'])
        
        self.training_steps = checkpoint.get('training_steps', 0)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'type': f'ExpertApprentice ({self.mode})',
            'mode': self.mode,
            'hidden_dims': self.hidden_dims,
        })
        return config


# Test
if __name__ == "__main__":
    print("=== Test de Expert Apprentice ===\n")
    
    from deeprl.envs.tictactoe import TicTacToeVsRandom
    
    env = TicTacToeVsRandom()
    
    # Créer un expert MCTS
    print("1. Création de l'expert MCTS:")
    expert = MCTSExpert(n_simulations=100)
    print(f"   Expert créé avec 100 simulations")
    
    # Tester l'expert
    state = env.reset()
    action = expert.get_action(state, env, env.get_available_actions())
    print(f"   Action de l'expert: {action}")
    
    # Behavior Cloning
    print("\n2. Behavior Cloning:")
    bc_agent = ExpertApprenticeAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        expert=expert,
        mode="bc",
        hidden_dims=[64, 64]
    )
    print(f"   Agent: {bc_agent.name}")
    
    # Entraînement court
    print("   Entraînement...")
    bc_agent.train(
        env, n_iterations=2,
        episodes_per_iteration=5,
        epochs_per_iteration=10,
        verbose=False
    )
    
    action = bc_agent.act(state, env.get_available_actions())
    print(f"   Action après entraînement: {action}")
    
    # DAgger
    print("\n3. DAgger:")
    dagger_agent = ExpertApprenticeAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        expert=expert,
        mode="dagger",
        hidden_dims=[64, 64]
    )
    print(f"   Agent: {dagger_agent.name}")
    
    # Entraînement court
    print("   Entraînement...")
    dagger_agent.train(
        env, n_iterations=2,
        episodes_per_iteration=5,
        epochs_per_iteration=10,
        verbose=False
    )
    
    action = dagger_agent.act(state, env.get_available_actions())
    print(f"   Action après entraînement: {action}")
    
    print("\n[OK] Tests passes!")
