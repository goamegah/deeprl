"""
Benchmark - Systeme complet de benchmarking des agents.

Ce module permet de:
1. Entrainer les agents sur differents checkpoints (1K, 10K, 100K, 1M episodes)
2. Mesurer les metriques demandees:
   - Score moyen par agent
   - Longueur moyenne des episodes
   - Temps moyen par action
3. Generer des courbes d'apprentissage
4. Comparer les agents sur un meme environnement
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Type
from dataclasses import dataclass, field
import numpy as np

from deeprl.envs.base import Environment
from deeprl.agents.base import Agent


@dataclass
class AgentBenchmarkResult:
    """Resultats de benchmark pour un agent."""
    
    agent_name: str
    agent_config: Dict[str, Any]
    
    # Resultats par checkpoint (n_episodes -> metrics)
    checkpoint_results: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Courbes d'apprentissage
    training_rewards: List[float] = field(default_factory=list)
    training_lengths: List[float] = field(default_factory=list)
    
    # Episodes correspondants pour les courbes
    training_episodes: List[int] = field(default_factory=list)
    
    def add_checkpoint(
        self,
        n_episodes: int,
        mean_score: float,
        std_score: float,
        mean_length: float,
        std_length: float,
        mean_action_time: float,
        win_rate: Optional[float] = None
    ):
        """Ajoute un point de mesure au checkpoint."""
        self.checkpoint_results[n_episodes] = {
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_length': mean_length,
            'std_length': std_length,
            'mean_action_time': mean_action_time,
            'win_rate': win_rate
        }
    
    def add_training_point(
        self,
        episode: int,
        reward: float,
        length: float
    ):
        """Ajoute un point sur la courbe d'apprentissage."""
        self.training_episodes.append(episode)
        self.training_rewards.append(reward)
        self.training_lengths.append(length)


@dataclass
class BenchmarkSuite:
    """
    Suite complete de benchmark.
    
    Gere le benchmark de plusieurs agents sur un environnement.
    """
    
    env_name: str
    results: Dict[str, AgentBenchmarkResult] = field(default_factory=dict)
    
    # Checkpoints par defaut
    checkpoints: List[int] = field(default_factory=lambda: [1000, 10000, 100000])
    
    def add_agent_result(self, result: AgentBenchmarkResult):
        """Ajoute les resultats d'un agent."""
        self.results[result.agent_name] = result
    
    def get_comparison_table(self) -> str:
        """
        Genere une table de comparaison formatee.
        
        Returns:
            String contenant la table formatee
        """
        lines = []
        lines.append("=" * 90)
        lines.append(f"BENCHMARK RESULTS - {self.env_name}")
        lines.append("=" * 90)
        
        # En-tete
        header = f"{'Agent':<25} {'Checkpoint':<12} {'Score':<20} {'Length':<15} {'Time/action':<12}"
        lines.append(header)
        lines.append("-" * 90)
        
        for agent_name, result in self.results.items():
            first_line = True
            for checkpoint in sorted(result.checkpoint_results.keys()):
                metrics = result.checkpoint_results[checkpoint]
                
                name_col = agent_name if first_line else ""
                score = f"{metrics['mean_score']:.3f} +/- {metrics['std_score']:.3f}"
                length = f"{metrics['mean_length']:.1f}"
                time_ms = f"{metrics['mean_action_time']*1000:.3f}ms"
                
                lines.append(f"{name_col:<25} {checkpoint:<12} {score:<20} {length:<15} {time_ms:<12}")
                first_line = False
            lines.append("-" * 90)
        
        return "\n".join(lines)
    
    def get_csv_data(self) -> str:
        """
        Exporte les resultats en format CSV.
        
        Returns:
            String CSV
        """
        lines = ["agent,checkpoint,mean_score,std_score,mean_length,std_length,mean_action_time,win_rate"]
        
        for agent_name, result in self.results.items():
            for checkpoint in sorted(result.checkpoint_results.keys()):
                m = result.checkpoint_results[checkpoint]
                win_rate = m['win_rate'] if m['win_rate'] is not None else ""
                lines.append(
                    f"{agent_name},{checkpoint},{m['mean_score']:.4f},{m['std_score']:.4f},"
                    f"{m['mean_length']:.2f},{m['std_length']:.2f},{m['mean_action_time']:.6f},{win_rate}"
                )
        
        return "\n".join(lines)


class Benchmark:
    """
    Classe principale pour executer des benchmarks.
    
    Usage:
        benchmark = Benchmark(env, checkpoints=[1000, 10000, 100000])
        benchmark.add_agent("DQN", DQNAgent, agent_kwargs)
        benchmark.add_agent("Q-Learning", QLearning, agent_kwargs)
        suite = benchmark.run(eval_episodes=100)
        benchmark.plot_results(suite, save_path="results.png")
    """
    
    def __init__(
        self,
        env: Environment,
        checkpoints: List[int] = None,
        eval_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        log_interval: int = 100,
        verbose: bool = True
    ):
        """
        Initialise le benchmark.
        
        Args:
            env: Environnement de base
            checkpoints: Points de mesure (defaut: [1000, 10000, 100000])
            eval_episodes: Episodes d'evaluation par checkpoint
            max_steps_per_episode: Limite de steps
            log_interval: Intervalle de log pour courbes
            verbose: Afficher la progression
        """
        self.env = env
        self.checkpoints = checkpoints or [1000, 10000, 100000]
        self.eval_episodes = eval_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.log_interval = log_interval
        self.verbose = verbose
        
        # Agents a benchmarker: (name, agent_class, kwargs)
        self.agents: List[Tuple[str, Type[Agent], Dict]] = []
    
    def add_agent(
        self,
        name: str,
        agent_class: Type[Agent],
        agent_kwargs: Dict[str, Any] = None
    ):
        """
        Ajoute un agent au benchmark.
        
        Args:
            name: Nom de l'agent
            agent_class: Classe de l'agent
            agent_kwargs: Arguments pour instancier l'agent
        """
        self.agents.append((name, agent_class, agent_kwargs or {}))
    
    def run(self) -> BenchmarkSuite:
        """
        Execute le benchmark complet.
        
        Returns:
            BenchmarkSuite avec tous les resultats
        """
        suite = BenchmarkSuite(
            env_name=self.env.name,
            checkpoints=self.checkpoints
        )
        
        for agent_name, agent_class, agent_kwargs in self.agents:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking: {agent_name}")
                print(f"{'='*60}")
            
            result = self._benchmark_agent(agent_name, agent_class, agent_kwargs)
            suite.add_agent_result(result)
        
        return suite
    
    def _benchmark_agent(
        self,
        agent_name: str,
        agent_class: Type[Agent],
        agent_kwargs: Dict[str, Any]
    ) -> AgentBenchmarkResult:
        """
        Benchmark un seul agent.
        
        Args:
            agent_name: Nom de l'agent
            agent_class: Classe de l'agent
            agent_kwargs: Arguments d'instanciation
        
        Returns:
            AgentBenchmarkResult
        """
        # Creer l'agent
        agent = agent_class(**agent_kwargs)
        
        result = AgentBenchmarkResult(
            agent_name=agent_name,
            agent_config=agent_kwargs
        )
        
        total_episodes = 0
        sorted_checkpoints = sorted(self.checkpoints)
        
        for checkpoint in sorted_checkpoints:
            episodes_to_train = checkpoint - total_episodes
            
            if self.verbose:
                print(f"\n  Training {episodes_to_train} episodes (total: {checkpoint})...")
            
            # Entrainer
            self._train_agent(
                agent, episodes_to_train, result, total_episodes
            )
            total_episodes = checkpoint
            
            if self.verbose:
                print(f"  Evaluating ({self.eval_episodes} episodes)...")
            
            # Evaluer
            eval_metrics = self._evaluate_agent(agent)
            
            result.add_checkpoint(
                n_episodes=checkpoint,
                **eval_metrics
            )
            
            if self.verbose:
                print(f"    Score: {eval_metrics['mean_score']:.3f} +/- {eval_metrics['std_score']:.3f}")
                print(f"    Length: {eval_metrics['mean_length']:.1f}")
                print(f"    Time/action: {eval_metrics['mean_action_time']*1000:.3f}ms")
        
        return result
    
    def _train_agent(
        self,
        agent: Agent,
        n_episodes: int,
        result: AgentBenchmarkResult,
        episode_offset: int
    ) -> Dict:
        """
        Entraine un agent et collecte les metriques d'apprentissage.
        """
        rewards_buffer = []
        lengths_buffer = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            agent.on_episode_start()
            
            total_reward = 0.0
            steps = 0
            
            while not self.env.is_game_over and steps < self.max_steps_per_episode:
                available = self.env.get_available_actions()
                action = agent.act(state, available, training=True, env=self.env)
                
                next_state, reward, done = self.env.step(action)
                
                agent.learn(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                state = next_state
                total_reward += reward
                steps += 1
            
            agent.on_episode_end(total_reward, steps)
            
            rewards_buffer.append(total_reward)
            lengths_buffer.append(steps)
            
            # Logger a intervalles reguliers
            if (episode + 1) % self.log_interval == 0:
                mean_reward = np.mean(rewards_buffer[-self.log_interval:])
                mean_length = np.mean(lengths_buffer[-self.log_interval:])
                
                result.add_training_point(
                    episode=episode_offset + episode + 1,
                    reward=mean_reward,
                    length=mean_length
                )
        
        return {
            'mean_reward': np.mean(rewards_buffer),
            'mean_length': np.mean(lengths_buffer)
        }
    
    def _evaluate_agent(self, agent: Agent) -> Dict[str, float]:
        """
        Evalue un agent en mode exploitation.
        
        Returns:
            Dictionnaire avec toutes les metriques
        """
        agent.set_training_mode(False)
        
        scores = []
        lengths = []
        action_times = []
        wins = 0
        losses = 0
        draws = 0
        
        for _ in range(self.eval_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            episode_action_times = []
            
            while not self.env.is_game_over and steps < self.max_steps_per_episode:
                available = self.env.get_available_actions()
                
                start_time = time.time()
                action = agent.act(state, available, training=False, env=self.env)
                action_time = time.time() - start_time
                episode_action_times.append(action_time)
                
                state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
            
            scores.append(total_reward)
            lengths.append(steps)
            action_times.extend(episode_action_times)
            
            # Determiner l'issue
            if total_reward > 0:
                wins += 1
            elif total_reward < 0:
                losses += 1
            else:
                draws += 1
        
        agent.set_training_mode(True)
        
        n = len(scores)
        win_rate = wins / n if n > 0 else None
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'mean_action_time': np.mean(action_times) if action_times else 0,
            'win_rate': win_rate
        }


def quick_benchmark(
    env: Environment,
    agents: Dict[str, Agent],
    checkpoints: List[int] = None,
    eval_episodes: int = 100,
    verbose: bool = True
) -> BenchmarkSuite:
    """
    Fonction utilitaire pour un benchmark rapide.
    
    Args:
        env: Environnement
        agents: Dictionnaire {nom: agent}
        checkpoints: Points de mesure
        eval_episodes: Episodes d'evaluation
        verbose: Afficher progression
    
    Returns:
        BenchmarkSuite
    
    Example:
        >>> from deeprl import GridWorld, RandomAgent, TabularQLearning
        >>> env = GridWorld.create_simple(5)
        >>> agents = {
        ...     'Random': RandomAgent(state_dim=25, n_actions=4),
        ...     'Q-Learning': TabularQLearning(n_states=25, n_actions=4)
        ... }
        >>> suite = quick_benchmark(env, agents, checkpoints=[1000, 5000])
    """
    checkpoints = checkpoints or [1000, 10000]
    suite = BenchmarkSuite(env_name=env.name, checkpoints=checkpoints)
    
    for agent_name, agent in agents.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {agent_name}")
            print(f"{'='*60}")
        
        result = AgentBenchmarkResult(
            agent_name=agent_name,
            agent_config=agent.get_config() if hasattr(agent, 'get_config') else {}
        )
        
        total_episodes = 0
        
        for checkpoint in sorted(checkpoints):
            episodes_to_train = checkpoint - total_episodes
            
            if verbose:
                print(f"\n  Training {episodes_to_train} episodes...")
            
            # Train
            for ep in range(episodes_to_train):
                state = env.reset()
                agent.on_episode_start()
                
                total_reward = 0
                steps = 0
                
                while not env.is_game_over and steps < 1000:
                    available = env.get_available_actions()
                    action = agent.act(state, available, training=True, env=env)
                    next_state, reward, done = env.step(action)
                    agent.learn(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                agent.on_episode_end(total_reward, steps)
                
                if (ep + 1) % 100 == 0:
                    result.add_training_point(total_episodes + ep + 1, total_reward, steps)
            
            total_episodes = checkpoint
            
            # Evaluate
            if verbose:
                print(f"  Evaluating...")
            
            agent.set_training_mode(False)
            scores = []
            lengths = []
            action_times = []
            
            for _ in range(eval_episodes):
                state = env.reset()
                total_reward = 0
                steps = 0
                
                while not env.is_game_over and steps < 1000:
                    available = env.get_available_actions()
                    start = time.time()
                    action = agent.act(state, available, training=False, env=env)
                    action_times.append(time.time() - start)
                    
                    state, reward, done = env.step(action)
                    total_reward += reward
                    steps += 1
                
                scores.append(total_reward)
                lengths.append(steps)
            
            agent.set_training_mode(True)
            
            result.add_checkpoint(
                n_episodes=checkpoint,
                mean_score=np.mean(scores),
                std_score=np.std(scores),
                mean_length=np.mean(lengths),
                std_length=np.std(lengths),
                mean_action_time=np.mean(action_times)
            )
            
            if verbose:
                print(f"    Score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
        
        suite.add_agent_result(result)
    
    # Afficher la table
    if verbose:
        print("\n" + suite.get_comparison_table())
    
    return suite


if __name__ == "__main__":
    # Test du module benchmark
    print("=== Test du module Benchmark ===\n")
    
    from deeprl.envs.grid_world import GridWorld
    from deeprl.agents.random_agent import RandomAgent
    from deeprl.agents.tabular.q_learning import TabularQLearning
    
    env = GridWorld.create_simple(size=5)
    
    agents = {
        'Random': RandomAgent(state_dim=25, n_actions=4),
        'Q-Learning': TabularQLearning(n_states=25, n_actions=4, lr=0.1, gamma=0.99)
    }
    
    suite = quick_benchmark(
        env=env,
        agents=agents,
        checkpoints=[500, 1000],
        eval_episodes=50,
        verbose=True
    )
    
    # Afficher CSV
    print("\n" + suite.get_csv_data())
    
    print("\nTest termine!")
