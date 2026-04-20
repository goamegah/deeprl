"""
Evaluator - Évaluation des agents entraînés.

Ce module permet d'évaluer les performances d'un agent APRÈS l'entraînement.
C'est différent des métriques d'entraînement car:
- L'agent est en mode évaluation (pas d'exploration)
- On mesure les vraies performances de la politique apprise
- On peut comparer différents agents de manière équitable

Métriques calculées:
- Score moyen sur N parties
- Longueur moyenne des parties
- Temps moyen par action
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm

from deeprl.envs.base import Environment
from deeprl.agents.base import Agent


@dataclass
class EvaluationResults:
    """
    Résultats d'une évaluation.
    
    Stocke toutes les métriques demandées dans le projet.
    """
    agent_name: str
    n_episodes: int
    
    # Scores
    scores: List[float] = field(default_factory=list)
    
    # Longueurs
    episode_lengths: List[int] = field(default_factory=list)
    
    # Temps
    action_times: List[float] = field(default_factory=list)  # Temps par action
    episode_times: List[float] = field(default_factory=list)  # Temps par épisode
    
    # Victoires (pour les jeux)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    def add_episode(
        self,
        score: float,
        length: int,
        time_elapsed: float,
        action_times: List[float],
        outcome: Optional[str] = None  # "win", "loss", "draw"
    ):
        """Ajoute les résultats d'un épisode."""
        self.scores.append(score)
        self.episode_lengths.append(length)
        self.episode_times.append(time_elapsed)
        self.action_times.extend(action_times)
        
        if outcome == "win":
            self.wins += 1
        elif outcome == "loss":
            self.losses += 1
        elif outcome == "draw":
            self.draws += 1
    
    def get_summary(self) -> Dict[str, float]:
        """
        Retourne un résumé complet des métriques.
        
        Returns:
            Dictionnaire avec toutes les métriques demandées
        """
        if len(self.scores) == 0:
            return {}
        
        return {
            # Métriques de score
            "mean_score": np.mean(self.scores),
            "std_score": np.std(self.scores),
            "min_score": np.min(self.scores),
            "max_score": np.max(self.scores),
            "median_score": np.median(self.scores),
            
            # Métriques de longueur
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "min_length": np.min(self.episode_lengths),
            "max_length": np.max(self.episode_lengths),
            
            # Métriques de temps
            "mean_action_time": np.mean(self.action_times) if self.action_times else 0,
            "std_action_time": np.std(self.action_times) if self.action_times else 0,
            "mean_episode_time": np.mean(self.episode_times),
            "games_per_second": len(self.scores) / sum(self.episode_times) if sum(self.episode_times) > 0 else 0,
            
            # Métriques de victoire (pour les jeux)
            "win_rate": self.wins / len(self.scores) if len(self.scores) > 0 else 0,
            "loss_rate": self.losses / len(self.scores) if len(self.scores) > 0 else 0,
            "draw_rate": self.draws / len(self.scores) if len(self.scores) > 0 else 0,
            
            # Méta-informations
            "n_episodes": len(self.scores),
            "total_steps": sum(self.episode_lengths),
        }
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"EvaluationResults({self.agent_name}, n={self.n_episodes}):\n"
            f"  Score: {summary.get('mean_score', 0):.3f} ± {summary.get('std_score', 0):.3f}\n"
            f"  Length: {summary.get('mean_length', 0):.1f} ± {summary.get('std_length', 0):.1f}\n"
            f"  Time/action: {summary.get('mean_action_time', 0)*1000:.3f}ms"
        )


class Evaluator:
    """
    Classe pour évaluer les performances d'un agent.
    
    L'évaluation se fait avec l'agent en mode "exploitation pure"
    (pas d'exploration aléatoire).
    
    Exemple d'utilisation:
        >>> evaluator = Evaluator(env, agent)
        >>> results = evaluator.evaluate(n_episodes=100)
        >>> print(results.get_summary())
    """
    
    def __init__(
        self,
        env: Environment,
        agent: Agent,
        verbose: bool = True
    ):
        """
        Initialise l'évaluateur.
        
        Args:
            env: L'environnement d'évaluation
            agent: L'agent à évaluer
            verbose: Afficher la progression
        """
        self.env = env
        self.agent = agent
        self.verbose = verbose
    
    def evaluate(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        render: bool = False
    ) -> EvaluationResults:
        """
        Évalue l'agent sur N épisodes.
        
        Args:
            n_episodes: Nombre d'épisodes d'évaluation
            max_steps_per_episode: Limite de steps par épisode
            render: Afficher l'environnement à chaque step
        
        Returns:
            Les résultats d'évaluation
        """
        results = EvaluationResults(
            agent_name=self.agent.name,
            n_episodes=n_episodes
        )
        
        # Mettre l'agent en mode évaluation
        self.agent.set_training_mode(False)
        
        # Barre de progression
        pbar = tqdm(
            range(n_episodes),
            desc=f"Evaluating {self.agent.name}",
            disable=not self.verbose
        )
        
        for episode in pbar:
            episode_result = self._run_episode(max_steps_per_episode, render)
            
            results.add_episode(
                score=episode_result["total_reward"],
                length=episode_result["steps"],
                time_elapsed=episode_result["time"],
                action_times=episode_result["action_times"],
                outcome=episode_result.get("outcome")
            )
            
            # Mise à jour de la barre de progression
            if (episode + 1) % 10 == 0:
                summary = results.get_summary()
                pbar.set_postfix({
                    "score": f"{summary['mean_score']:.2f}",
                    "length": f"{summary['mean_length']:.1f}"
                })
        
        # Remettre l'agent en mode entraînement
        self.agent.set_training_mode(True)
        
        return results
    
    def _run_episode(self, max_steps: int, render: bool) -> Dict:
        """
        Exécute un seul épisode d'évaluation.
        
        Args:
            max_steps: Nombre maximum de steps
            render: Afficher l'environnement
        
        Returns:
            Dictionnaire avec les résultats de l'épisode
        """
        state = self.env.reset()
        
        total_reward = 0.0
        steps = 0
        action_times = []
        
        start_time = time.time()
        
        if render:
            self.env.render()
        
        while not self.env.is_game_over and steps < max_steps:
            # Mesurer le temps de décision
            action_start = time.time()
            
            available_actions = self.env.get_available_actions()
            action = self.agent.act(
                state,
                available_actions=available_actions,
                training=False,  # Mode évaluation
                env=self.env  # Pour MCTS/AlphaZero qui ont besoin de simuler
            )
            
            action_time = time.time() - action_start
            action_times.append(action_time)
            
            # Exécuter l'action
            next_state, reward, done = self.env.step(action)
            
            if render:
                self.env.render()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        elapsed_time = time.time() - start_time
        
        # Déterminer l'issue (pour les jeux)
        outcome = self._determine_outcome(total_reward)
        
        return {
            "total_reward": total_reward,
            "steps": steps,
            "time": elapsed_time,
            "action_times": action_times,
            "outcome": outcome
        }
    
    def _determine_outcome(self, total_reward: float) -> Optional[str]:
        """
        Détermine l'issue de la partie basée sur la récompense.
        
        Pour des jeux, on considère généralement:
        - reward > 0: victoire
        - reward < 0: défaite
        - reward == 0: match nul
        
        Returns:
            "win", "loss", "draw", ou None
        """
        # Cette logique peut être personnalisée selon l'environnement
        if total_reward > 0:
            return "win"
        elif total_reward < 0:
            return "loss"
        else:
            return "draw"


# Test rapide si exécuté directement
if __name__ == "__main__":
    from deeprl.envs.line_world import LineWorld
    from deeprl.agents.random_agent import RandomAgent
    
    print("=== Test de l'Evaluator ===\n")
    
    # Créer l'environnement et l'agent
    env = LineWorld(size=5)
    agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    # Évaluer
    evaluator = Evaluator(env, agent, verbose=True)
    results = evaluator.evaluate(n_episodes=100)
    
    # Afficher les résultats
    print("\n=== Résultats d'évaluation ===")
    print(results)
    
    print("\n=== Détails ===")
    summary = results.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n[OK] Test passe!")
