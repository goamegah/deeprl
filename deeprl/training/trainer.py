"""
Trainer - Boucle d'entraînement pour les agents.

Ce module gère la boucle d'entraînement standard:
1. Reset de l'environnement
2. L'agent choisit une action
3. L'environnement exécute l'action
4. L'agent apprend de la transition
5. Répéter jusqu'à la fin de l'épisode
6. Répéter pour N épisodes
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from deeprl.envs.base import Environment
from deeprl.agents.base import Agent


@dataclass
class TrainingMetrics:
    """
    Métriques collectées pendant l'entraînement.
    
    Attributes:
        episode_rewards: Récompense totale par épisode
        episode_lengths: Longueur de chaque épisode
        episode_times: Temps d'exécution par épisode
        learning_metrics: Métriques d'apprentissage (loss, etc.)
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_times: List[float] = field(default_factory=list)
    learning_metrics: List[Dict[str, float]] = field(default_factory=list)
    
    def add_episode(
        self,
        reward: float,
        length: int,
        time_elapsed: float,
        learning_info: Optional[Dict[str, float]] = None
    ):
        """Ajoute les métriques d'un épisode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_times.append(time_elapsed)
        if learning_info:
            self.learning_metrics.append(learning_info)
    
    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """
        Retourne un résumé des N derniers épisodes.
        
        Args:
            last_n: Nombre d'épisodes à considérer
        
        Returns:
            Dictionnaire avec moyennes et écarts-types
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        rewards = self.episode_rewards[-last_n:]
        lengths = self.episode_lengths[-last_n:]
        times = self.episode_times[-last_n:]
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "mean_time": np.mean(times),
            "total_episodes": len(self.episode_rewards),
        }


class Trainer:
    """
    Classe pour entraîner un agent sur un environnement.
    
    Gère la boucle d'entraînement complète avec:
    - Logging des métriques
    - Callbacks personnalisés
    - Affichage de progression
    
    Exemple d'utilisation:
        >>> trainer = Trainer(env, agent)
        >>> metrics = trainer.train(n_episodes=1000)
        >>> print(metrics.get_summary())
    """
    
    def __init__(
        self,
        env: Environment,
        agent: Agent,
        verbose: bool = True,
        log_interval: int = 100
    ):
        """
        Initialise le trainer.
        
        Args:
            env: L'environnement d'entraînement
            agent: L'agent à entraîner
            verbose: Afficher les logs de progression
            log_interval: Intervalle entre les logs (en épisodes)
        """
        self.env = env
        self.agent = agent
        self.verbose = verbose
        self.log_interval = log_interval
        
        # Métriques
        self.metrics = TrainingMetrics()
    
    def train(
        self,
        n_episodes: int,
        max_steps_per_episode: int = 1000,
        callbacks: Optional[List[Callable]] = None
    ) -> TrainingMetrics:
        """
        Entraîne l'agent sur N épisodes.
        
        Args:
            n_episodes: Nombre d'épisodes d'entraînement
            max_steps_per_episode: Limite de steps par épisode
            callbacks: Fonctions appelées à chaque épisode
        
        Returns:
            Les métriques d'entraînement
        """
        callbacks = callbacks or []
        
        # Barre de progression
        pbar = tqdm(
            range(n_episodes),
            desc=f"Training {self.agent.name}",
            disable=not self.verbose
        )
        
        for episode in pbar:
            episode_metrics = self._run_episode(max_steps_per_episode)
            
            # Ajouter les métriques
            self.metrics.add_episode(
                reward=episode_metrics["total_reward"],
                length=episode_metrics["steps"],
                time_elapsed=episode_metrics["time"],
                learning_info=episode_metrics.get("learning_info")
            )
            
            # Appeler les callbacks
            for callback in callbacks:
                callback(episode, episode_metrics, self.metrics)
            
            # Mise à jour de la barre de progression
            if (episode + 1) % self.log_interval == 0 or episode == n_episodes - 1:
                summary = self.metrics.get_summary(last_n=self.log_interval)
                pbar.set_postfix({
                    "reward": f"{summary['mean_reward']:.2f}",
                    "length": f"{summary['mean_length']:.1f}"
                })
        
        return self.metrics
    
    def _run_episode(self, max_steps: int) -> Dict:
        """
        Exécute un seul épisode d'entraînement.
        
        Args:
            max_steps: Nombre maximum de steps
        
        Returns:
            Dictionnaire avec les métriques de l'épisode
        """
        # Initialisation
        state = self.env.reset()
        self.agent.on_episode_start()
        
        total_reward = 0.0
        steps = 0
        episode_learning_info = {}
        
        start_time = time.time()
        
        # Boucle de l'épisode
        while not self.env.is_game_over and steps < max_steps:
            # L'agent choisit une action
            available_actions = self.env.get_available_actions()
            action = self.agent.act(
                state,
                available_actions=available_actions,
                training=True,
                env=self.env  # Pour MCTS/AlphaZero qui ont besoin de simuler
            )
            
            # L'environnement exécute l'action
            next_state, reward, done = self.env.step(action)
            
            # L'agent apprend
            learning_info = self.agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                available_actions=available_actions,
                available_actions_next=self.env.get_available_actions() if not done else []
            )
            
            # Accumuler les métriques d'apprentissage
            if learning_info:
                for key, value in learning_info.items():
                    if key not in episode_learning_info:
                        episode_learning_info[key] = []
                    episode_learning_info[key].append(value)
            
            # Mettre à jour pour le prochain step
            state = next_state
            total_reward += reward
            steps += 1
        
        elapsed_time = time.time() - start_time
        
        # Notifier l'agent de la fin de l'épisode
        self.agent.on_episode_end(total_reward, steps)
        
        # Moyenner les métriques d'apprentissage
        avg_learning_info = {
            key: np.mean(values)
            for key, values in episode_learning_info.items()
        }
        
        return {
            "total_reward": total_reward,
            "steps": steps,
            "time": elapsed_time,
            "learning_info": avg_learning_info if avg_learning_info else None
        }
    
    def reset_metrics(self):
        """Réinitialise les métriques d'entraînement."""
        self.metrics = TrainingMetrics()


# Test rapide si exécuté directement
if __name__ == "__main__":
    from deeprl.envs.line_world import LineWorld
    from deeprl.agents.random_agent import RandomAgent
    
    print("=== Test du Trainer ===\n")
    
    # Créer l'environnement et l'agent
    env = LineWorld(size=5)
    agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    # Créer le trainer
    trainer = Trainer(env, agent, verbose=True, log_interval=100)
    
    # Entraîner
    metrics = trainer.train(n_episodes=500)
    
    # Afficher les résultats
    print("\n=== Résultats ===")
    summary = metrics.get_summary()
    print(f"  Épisodes: {summary['total_episodes']}")
    print(f"  Récompense moyenne: {summary['mean_reward']:.3f} ± {summary['std_reward']:.3f}")
    print(f"  Longueur moyenne: {summary['mean_length']:.1f} ± {summary['std_length']:.1f}")
    
    print("\n[OK] Test passe!")
