"""
DeepRL - Point d'entree principal

Ce fichier permet de:
1. Tester rapidement les environnements et agents
2. Lancer des benchmarks complets avec graphiques
3. Jouer contre un agent IA
4. Demontrer l'utilisation de la bibliotheque

Usage:
    python main.py                    # Test rapide (GridWorld + Q-Learning)
    python main.py --benchmark        # Benchmark complet avec graphiques
    python main.py --gui              # Observer un agent jouer
    python main.py --play             # Jouer contre un agent IA
    python main.py --env lineworld    # Environnement specifique
    python main.py --env gridworld    # GridWorld avec Q-Learning
    python main.py --env tictactoe    # TicTacToe avec DQN
    python main.py --env reinforce    # GridWorld avec REINFORCE
    python main.py --env ppo          # GridWorld avec PPO
    python main.py --env mcts         # TicTacToe avec MCTS
    python main.py --env alphazero    # TicTacToe avec AlphaZero
    python main.py --env muzero       # GridWorld avec MuZero
    python main.py --env stochastic-muzero  # GridWorld avec MuZero Stochastique
    python main.py --env quarto       # Quarto avec MCTS
    python main.py --env imitation    # Expert Apprentice (Behavior Cloning / DAgger)
"""

import argparse
import os
import sys
import numpy as np

from deeprl.envs import LineWorld, GridWorld, TicTacToe, TicTacToeVsRandom, Quarto
from deeprl.agents import (
    RandomAgent, TabularQLearning, DQNAgent,
    REINFORCEAgent, PPOAgent, MCTSAgent, RandomRolloutAgent,
    AlphaZeroAgent, MuZeroAgent, StochasticMuZeroAgent,
    ExpertApprenticeAgent, MCTSExpert
)
from deeprl.training import Trainer, Evaluator

# ============================================================================
# REGISTRY : pour chaque (env, agent_name), comment recréer l'agent
# ============================================================================

AGENT_REGISTRY = {
    "lineworld": {
        "Random":                   lambda: RandomAgent(state_dim=7, n_actions=2),
        "TabularQLearning":         lambda: TabularQLearning(n_states=7, n_actions=2, lr=0.1, gamma=0.99)
    },
    "gridworld": {
        "Random":                   lambda: RandomAgent(state_dim=25, n_actions=4),
        "TabularQLearning":         lambda: TabularQLearning(n_states=25, n_actions=4, lr=0.1, gamma=0.99)
    },
    "tictactoe": {
        "Random":                   lambda: RandomAgent(state_dim=27, n_actions=9)
    },
    "quarto": {
        "Random":                   lambda: RandomAgent(state_dim=114, n_actions=16)
    },
}

# Agents qui n'ont pas besoin d'entraînement (pas de .pt à charger)
NO_TRAINING_AGENTS = {"Random", "RandomRollout", "MCTS"}

# Agents qui ont besoin du param env= dans act()
NEEDS_ENV_AGENTS = {"MCTS", "RandomRollout", "AlphaZero", "MuZero", "MuZeroStochastic"}

# Agent par défaut pour chaque env (utilisé si --agent non spécifié)
DEFAULT_AGENT = {
    "lineworld": "TabularQLearning",
    "gridworld": "TabularQLearning",
    "tictactoe": "MCTS",
    "quarto": "MCTS",
}


def demo_lineworld():
    """
    Démonstration simple sur LineWorld.
    
    Montre le cycle complet:
    1. Créer environnement
    2. Créer agent
    3. Entraîner
    4. Évaluer
    """
    print("=" * 60)
    print("DEMO: LineWorld avec RandomAgent")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = LineWorld(size=5)
    print(f"\nEnvironnement: {env}")
    print(f"   - State shape: {env.state_shape}")
    print(f"   - Nombre d'actions: {env.n_actions}")
    
    # 2. Créer l'agent
    agent = RandomAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions
    )
    print(f"\nAgent: {agent}")
    
    # 3. Montrer un episode
    print("\nExemple d'episode:")
    state = env.reset()
    env.render()
    
    step = 0
    total_reward = 0
    while not env.is_game_over and step < 20:
        action = agent.act(state, env.get_available_actions())
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        
        action_name = "←" if action == 0 else "→"
        print(f"Step {step}: Action={action_name}, Reward={reward:.2f}")
        env.render()
    
    print(f"\n[OK] Episode termine en {step} steps, reward total: {total_reward:.2f}")
    
    # 4. Entrainer (meme si RandomAgent n'apprend pas)
    print("\n" + "=" * 60)
    print("ENTRAINEMENT (500 episodes)")
    print("=" * 60)
    
    trainer = Trainer(env, agent, verbose=True, log_interval=100)
    metrics = trainer.train(n_episodes=500)
    
    summary = metrics.get_summary()
    print(f"\nResultats d'entrainement:")
    print(f"   - Recompense moyenne: {summary['mean_reward']:.3f}")
    print(f"   - Longueur moyenne: {summary['mean_length']:.1f}")
    
    # 5. Evaluer
    print("\n" + "=" * 60)
    print("EVALUATION (100 episodes)")
    print("=" * 60)
    
    evaluator = Evaluator(env, agent, verbose=True)
    results = evaluator.evaluate(n_episodes=100)
    
    eval_summary = results.get_summary()
    print(f"\nResultats d'evaluation:")
    print(f"   - Score moyen: {eval_summary['mean_score']:.3f} +/- {eval_summary['std_score']:.3f}")
    print(f"   - Longueur moyenne: {eval_summary['mean_length']:.1f} +/- {eval_summary['std_length']:.1f}")
    print(f"   - Temps/action: {eval_summary['mean_action_time']*1000:.4f} ms")


def demo_gridworld():
    """
    Démonstration de GridWorld avec Q-Learning.
    
    Montre comment un agent APPREND vraiment à résoudre le problème.
    Compare Random vs Q-Learning.
    """
    print("=" * 60)
    print("DEMO: GridWorld avec TabularQLearning")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = GridWorld.create_simple(size=5)
    print(f"\nEnvironnement: {env}")
    print(f"   - State shape: {env.state_shape}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")
    print(f"   - Chemin optimal: {env.get_optimal_path_length()} steps")
    
    env.reset()
    print("\nGrille initiale:")
    env.render()
    
    # 2. Comparer Random vs Q-Learning
    print("\n" + "=" * 60)
    print("COMPARAISON: Random vs Q-Learning")
    print("=" * 60)
    
    # Agent Random
    print("\n--- Agent Random ---")
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    trainer_random = Trainer(env, random_agent, verbose=True, log_interval=200)
    trainer_random.train(n_episodes=1000)
    
    evaluator_random = Evaluator(env, random_agent, verbose=False)
    results_random = evaluator_random.evaluate(n_episodes=100)
    
    # Agent Q-Learning
    print("\n--- Agent Q-Learning ---")
    q_agent = TabularQLearning(
        n_states=env.state_dim,  # 25 états pour grille 5x5
        n_actions=env.n_actions,
        lr=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    trainer_q = Trainer(env, q_agent, verbose=True, log_interval=200)
    trainer_q.train(n_episodes=1000)
    
    evaluator_q = Evaluator(env, q_agent, verbose=False)
    results_q = evaluator_q.evaluate(n_episodes=100)
    
    # 3. Afficher les resultats comparatifs
    print("\n" + "=" * 60)
    print("RESULTATS COMPARATIFS")
    print("=" * 60)
    
    summary_random = results_random.get_summary()
    summary_q = results_q.get_summary()
    optimal_length = env.get_optimal_path_length()
    
    print(f"\n{'Métrique':<25} {'Random':<20} {'Q-Learning':<20} {'Optimal':<10}")
    print("-" * 75)
    
    print(f"{'Score moyen':<25} {summary_random['mean_score']:.3f} ± {summary_random['std_score']:.3f}      {summary_q['mean_score']:.3f} ± {summary_q['std_score']:.3f}      -")
    print(f"{'Longueur moyenne':<25} {summary_random['mean_length']:.1f} ± {summary_random['std_length']:.1f}        {summary_q['mean_length']:.1f} ± {summary_q['std_length']:.1f}         {optimal_length}")
    print(f"{'Temps/action (ms)':<25} {summary_random['mean_action_time']*1000:.4f}              {summary_q['mean_action_time']*1000:.4f}             -")
    
    # 4. Visualiser la politique apprise
    print("\n" + "=" * 60)
    print("POLITIQUE APPRISE (Q-Learning)")
    print("=" * 60)
    
    visualize_policy(env, q_agent)
    
    # 5. Montrer un episode avec la politique apprise
    print("\n" + "=" * 60)
    print("EPISODE AVEC POLITIQUE APPRISE")
    print("=" * 60)
    
    q_agent.set_training_mode(False)  # Mode exploitation
    state = env.reset()
    env.render()
    
    total_reward = 0
    step = 0
    while not env.is_game_over and step < 50:
        action = q_agent.act(state, env.get_available_actions(), training=False)
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"Step {step}: {GridWorld.ACTION_NAMES[action]}, Reward={reward:.2f}")
        env.render()
    
    print(f"\n[OK] Episode termine en {step} steps (optimal: {optimal_length})")
    print(f"   Recompense totale: {total_reward:.2f}")


def visualize_policy(env: GridWorld, agent: TabularQLearning):
    """
    Affiche la politique apprise sous forme de flèches.
    """
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    policy = agent.get_policy()
    
    print("\nPolitique (meilleure action par état):")
    print("+" + "---+" * env.width)
    
    for row in range(env.height):
        line = "|"
        for col in range(env.width):
            pos = (row, col)
            state_idx = env.pos_to_index(pos)
            
            if pos == env.goal_pos:
                cell = " G "
            elif pos in env.walls:
                cell = " # "
            else:
                best_action = policy[state_idx]
                cell = f" {arrows[best_action]} "
            
            line += cell + "|"
        print(line)
        print("+" + "---+" * env.width)
    
    # Afficher aussi les valeurs V(s)
    print("\nValeurs V(s) (max Q pour chaque état):")
    values = agent.get_value_function()
    
    print("+" + "------+" * env.width)
    for row in range(env.height):
        line = "|"
        for col in range(env.width):
            pos = (row, col)
            state_idx = env.pos_to_index(pos)
            
            if pos == env.goal_pos:
                cell = "  G   "
            elif pos in env.walls:
                cell = "  #   "
            else:
                v = values[state_idx]
                cell = f"{v:+.2f} "
            
            line += cell + "|"
        print(line)
        print("+" + "------+" * env.width)


def demo_tictactoe():
    """
    Démonstration de TicTacToe avec DQN.
    
    Compare Random vs DQN (Deep Q-Network).
    """
    print("=" * 60)
    print("DEMO: TicTacToe avec Deep Q-Network")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = TicTacToeVsRandom(use_onehot=True)
    print(f"\nEnvironnement: {env}")
    print(f"   - State shape: {env.state_shape}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")
    
    # 2. Comparer Random vs DQN
    print("\n" + "=" * 60)
    print("COMPARAISON: Random vs DQN")
    print("=" * 60)
    
    n_train_episodes = 5000
    n_eval_episodes = 500
    
    # Agent Random
    print("\n--- Agent Random ---")
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    evaluator_random = Evaluator(env, random_agent, verbose=True)
    results_random = evaluator_random.evaluate(n_episodes=n_eval_episodes)
    
    # Agent DQN
    print("\n--- Agent DQN (entraînement) ---")
    dqn_agent = DQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[128, 128],
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        buffer_size=10000,
        batch_size=64,
        min_buffer_size=500,
        double_dqn=True,
        target_update_freq=100
    )
    
    trainer_dqn = Trainer(env, dqn_agent, verbose=True, log_interval=500)
    trainer_dqn.train(n_episodes=n_train_episodes)
    
    print("\n--- Agent DQN (évaluation) ---")
    evaluator_dqn = Evaluator(env, dqn_agent, verbose=True)
    results_dqn = evaluator_dqn.evaluate(n_episodes=n_eval_episodes)
    
    # 3. Afficher les resultats
    print("\n" + "=" * 60)
    print("RESULTATS COMPARATIFS")
    print("=" * 60)
    
    summary_random = results_random.get_summary()
    summary_dqn = results_dqn.get_summary()
    
    print(f"\n{'Métrique':<25} {'Random':<25} {'DQN':<25}")
    print("-" * 75)
    
    print(f"{'Score moyen':<25} {summary_random['mean_score']:.3f} ± {summary_random['std_score']:.3f}          {summary_dqn['mean_score']:.3f} ± {summary_dqn['std_score']:.3f}")
    print(f"{'Taux de victoire':<25} {summary_random['win_rate']*100:.1f}%                    {summary_dqn['win_rate']*100:.1f}%")
    print(f"{'Taux de défaite':<25} {summary_random['loss_rate']*100:.1f}%                    {summary_dqn['loss_rate']*100:.1f}%")
    print(f"{'Taux de nul':<25} {summary_random['draw_rate']*100:.1f}%                    {summary_dqn['draw_rate']*100:.1f}%")
    print(f"{'Longueur moyenne':<25} {summary_random['mean_length']:.1f}                      {summary_dqn['mean_length']:.1f}")
    
    # 4. Montrer une partie avec DQN
    print("\n" + "=" * 60)
    print("PARTIE AVEC DQN ENTRAINE")
    print("=" * 60)
    
    dqn_agent.set_training_mode(False)
    state = env.reset()
    env.render()
    
    while not env.is_game_over:
        available = env.get_available_actions()
        if len(available) == 0:
            break
        
        action = dqn_agent.act(state, available, training=False)
        state, reward, done = env.step(action)
        
        print(f"\nDQN joue position {action}:")
        env.render()
    
    print(f"\nRécompense finale: {reward}")


def run_benchmark():
    """
    Benchmark complet aux differents checkpoints avec generation de graphiques.
    
    Compare plusieurs agents sur GridWorld et TicTacToe.
    Genere des graphiques de resultats.
    """
    import os
    from deeprl.training.benchmark import quick_benchmark
    
    print("=" * 60)
    print("BENCHMARK COMPLET AVEC GRAPHIQUES")
    print("=" * 60)
    
    # Creer le dossier results
    os.makedirs("results", exist_ok=True)
    
    # =========================================
    # BENCHMARK 1: GridWorld
    # =========================================
    print("\n" + "=" * 60)
    print("BENCHMARK 1: GridWorld")
    print("=" * 60)
    
    env = GridWorld.create_simple(size=5)
    print(f"\nEnvironnement: {env}")
    print(f"Chemin optimal: {env.get_optimal_path_length()} steps")
    
    # Creer les agents a comparer
    agents_gridworld = {
        'Random': RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions),
        'Q-Learning': TabularQLearning(
            n_states=env.state_dim,
            n_actions=env.n_actions,
            lr=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.998
        ),
    }
    
    # Lancer le benchmark
    checkpoints = [1000, 5000, 10000]
    
    suite_gridworld = quick_benchmark(
        env=env,
        agents=agents_gridworld,
        checkpoints=checkpoints,
        eval_episodes=100,
        verbose=True
    )
    
    print("\n" + suite_gridworld.get_csv_data())
    
    # =========================================
    # BENCHMARK 2: TicTacToe
    # =========================================
    print("\n" + "=" * 60)
    print("BENCHMARK 2: TicTacToe")
    print("=" * 60)
    
    env_ttt = TicTacToeVsRandom(use_onehot=True)
    print(f"\nEnvironnement: {env_ttt}")
    
    agents_tictactoe = {
        'Random': RandomAgent(state_dim=env_ttt.state_dim, n_actions=env_ttt.n_actions),
        'DQN': DQNAgent(
            state_dim=env_ttt.state_dim,
            n_actions=env_ttt.n_actions,
            hidden_dims=[64, 64],
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.999,
            buffer_size=5000,
            batch_size=32,
            min_buffer_size=200,
            double_dqn=True,
            target_update_freq=100
        ),
    }
    
    suite_tictactoe = quick_benchmark(
        env=env_ttt,
        agents=agents_tictactoe,
        checkpoints=[500, 2000, 5000],
        eval_episodes=100,
        verbose=True
    )
    
    print("\n" + suite_tictactoe.get_csv_data())
    
    # =========================================
    # RESUME FINAL
    # =========================================
    print("\n" + "=" * 60)
    print("RESUME FINAL")
    print("=" * 60)
    
    print("\n--- GridWorld ---")
    print(suite_gridworld.get_comparison_table())
    
    print("\n--- TicTacToe ---")
    print(suite_tictactoe.get_comparison_table())
    
    print("\n" + "=" * 60)
    print("Graphiques generes:")
    print("  - results/benchmark_gridworld.png")
    print("  - results/benchmark_tictactoe.png")
    print("Donnees CSV:")
    print("  - results/benchmark_gridworld.csv")
    print("  - results/benchmark_tictactoe.csv")
    print("=" * 60)


def demo_reinforce():
    """
    Démonstration de REINFORCE sur GridWorld.
    
    Compare REINFORCE avec baseline vs sans baseline.
    """
    print("=" * 60)
    print("DEMO: REINFORCE (Policy Gradient)")
    print("=" * 60)
    
    # 1. Créer l'environnement
    env = GridWorld.create_simple(size=5)
    print(f"\n📦 Environnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Chemin optimal: {env.get_optimal_path_length()} steps")
    
    env.reset()
    env.render()
    
    n_episodes = 2000
    eval_episodes = 100
    
    # 2. REINFORCE sans baseline
    print("\n" + "=" * 60)
    print("REINFORCE (sans baseline)")
    print("=" * 60)
    
    agent_no_baseline = REINFORCEAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        gamma=0.99,
        baseline="none"
    )
    print(f"Agent: {agent_no_baseline.name}")
    
    trainer1 = Trainer(env, agent_no_baseline, verbose=True, log_interval=500)
    trainer1.train(n_episodes=n_episodes)
    
    evaluator1 = Evaluator(env, agent_no_baseline, verbose=True)
    results_no_baseline = evaluator1.evaluate(n_episodes=eval_episodes)
    
    # 3. REINFORCE avec baseline (Actor-Critic)
    print("\n" + "=" * 60)
    print("REINFORCE (avec Critic baseline)")
    print("=" * 60)
    
    agent_critic = REINFORCEAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        gamma=0.99,
        baseline="critic"
    )
    print(f"Agent: {agent_critic.name}")
    
    trainer2 = Trainer(env, agent_critic, verbose=True, log_interval=500)
    trainer2.train(n_episodes=n_episodes)
    
    evaluator2 = Evaluator(env, agent_critic, verbose=True)
    results_critic = evaluator2.evaluate(n_episodes=eval_episodes)
    
    # 4. Resultats comparatifs
    print("\n" + "=" * 60)
    print("RESULTATS COMPARATIFS")
    print("=" * 60)
    
    summary1 = results_no_baseline.get_summary()
    summary2 = results_critic.get_summary()
    optimal = env.get_optimal_path_length()
    
    print(f"\n{'Métrique':<25} {'Sans baseline':<20} {'Avec Critic':<20} {'Optimal':<10}")
    print("-" * 75)
    print(f"{'Score moyen':<25} {summary1['mean_score']:.3f}                {summary2['mean_score']:.3f}                -")
    print(f"{'Longueur moyenne':<25} {summary1['mean_length']:.1f}                 {summary2['mean_length']:.1f}                 {optimal}")
    
    # 5. Episode de demonstration
    print("\n" + "=" * 60)
    print("EPISODE AVEC REINFORCE+CRITIC")
    print("=" * 60)
    
    agent_critic.set_training_mode(False)
    state = env.reset()
    env.render()
    
    total_reward = 0
    step = 0
    while not env.is_game_over and step < 50:
        action = agent_critic.act(state, env.get_available_actions(), training=False)
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: {GridWorld.ACTION_NAMES[action]}")
        env.render()
    
    print(f"\n✅ Terminé en {step} steps (optimal: {optimal})")


def demo_ppo():
    """
    Démonstration de PPO sur GridWorld.
    """
    print("=" * 60)
    print("DEMO: PPO (Proximal Policy Optimization)")
    print("=" * 60)
    
    # 1. Créer l'environnement
    env = GridWorld.create_simple(size=5)
    print(f"\n📦 Environnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Chemin optimal: {env.get_optimal_path_length()} steps")
    
    env.reset()
    env.render()
    
    # 2. Creer l'agent PPO
    print("\n" + "=" * 60)
    print("Entrainement PPO")
    print("=" * 60)
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[64, 64],
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=4,
        batch_size=32,
        value_coef=0.5,
        entropy_coef=0.01
    )
    print(f"Agent: {agent.name}")
    print(f"Config: {agent.get_config()}")
    
    n_episodes = 2000
    
    trainer = Trainer(env, agent, verbose=True, log_interval=500)
    trainer.train(n_episodes=n_episodes)
    
# 3. Evaluation
    print("\n" + "=" * 60)
    print("Evaluation PPO")
    print("=" * 60)

    evaluator = Evaluator(env, agent, verbose=True)
    results = evaluator.evaluate(n_episodes=100)

    summary = results.get_summary()
    optimal = env.get_optimal_path_length()

    print(f"\nResultats PPO:")
    print(f"   Score moyen: {summary['mean_score']:.3f} +/- {summary['std_score']:.3f}")
    print(f"   Longueur moyenne: {summary['mean_length']:.1f} +/- {summary['std_length']:.1f}")
    print(f"   Chemin optimal: {optimal} steps")

    # 4. Episode de demonstration
    print("\n" + "=" * 60)
    print("EPISODE AVEC PPO")
    print("=" * 60)
    
    agent.set_training_mode(False)
    state = env.reset()
    env.render()
    
    step = 0
    while not env.is_game_over and step < 50:
        action = agent.act(state, env.get_available_actions(), training=False)
        state, reward, done = env.step(action)
        step += 1
        print(f"Step {step}: {GridWorld.ACTION_NAMES[action]}")
        env.render()
    
    print(f"\n[OK] Termine en {step} steps (optimal: {optimal})")


def demo_mcts():
    """
    Démonstration de MCTS sur TicTacToe.
    
    Compare Random, RandomRollout et MCTS.
    """
    print("=" * 60)
    print("DEMO: MCTS (Monte Carlo Tree Search)")
    print("=" * 60)

    # 1. Creer l'environnement
    env = TicTacToe()
    print(f"\nEnvironnement: {env}")
    # 2. Créer les agents
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    rollout_agent = RandomRolloutAgent(n_rollouts=20)
    mcts_agent = MCTSAgent(n_simulations=100, c_exploration=1.41)
    
    print(f"\nAgents:")
    print(f"   - {random_agent.name}")
    print(f"   - {rollout_agent.name}")
    print(f"   - {mcts_agent.name}")
    
    # 3. Fonction pour jouer une partie
    def play_game(env, player1, player2, p1_name, p2_name):
        """Joue une partie et retourne le gagnant."""
        state = env.reset()
        players = [player1, player2]
        
        while not env.is_game_over:
            current = env.current_player
            available = env.get_available_actions()
            
            if hasattr(players[current], 'n_simulations') or hasattr(players[current], 'n_rollouts'):
                action = players[current].act(state, available, env=env)
            else:
                action = players[current].act(state, available)
            
            state, reward, done = env.step(action)
        
        if env._winner == 0:
            return p1_name
        elif env._winner == 1:
            return p2_name
        else:
            return "Nul"
    
    # 4. Tournoi
    n_games = 20
    
    matchups = [
        (random_agent, rollout_agent, "Random", "Rollout"),
        (random_agent, mcts_agent, "Random", "MCTS"),
        (rollout_agent, mcts_agent, "Rollout", "MCTS"),
    ]
    
    print("\n" + "=" * 60)
    print("TOURNOI")
    print("=" * 60)
    
    for p1, p2, n1, n2 in matchups:
        wins = {n1: 0, n2: 0, "Nul": 0}
        
        for i in range(n_games):
            # Alterner qui commence
            if i % 2 == 0:
                winner = play_game(env, p1, p2, n1, n2)
            else:
                winner = play_game(env, p2, p1, n2, n1)
            wins[winner] += 1
        
        print(f"\n{n1} vs {n2} ({n_games} parties):")
        print(f"   {n1}: {wins[n1]} victoires ({wins[n1]/n_games*100:.0f}%)")
        print(f"   {n2}: {wins[n2]} victoires ({wins[n2]/n_games*100:.0f}%)")
        print(f"   Nuls: {wins['Nul']} ({wins['Nul']/n_games*100:.0f}%)")
    
    # 5. Partie de démonstration
    print("\n" + "=" * 60)
    print("📺 PARTIE MCTS vs RANDOM")
    print("=" * 60)
    
    state = env.reset()
    print("Début de partie:")
    env.render()
    
    while not env.is_game_over:
        available = env.get_available_actions()
        
        if env.current_player == 0:
            action = mcts_agent.act(state, available, env=env)
            player = "MCTS (X)"
        else:
            action = random_agent.act(state, available)
            player = "Random (O)"
        
        state, reward, done = env.step(action)
        print(f"\n{player} joue position {action}:")
        env.render()
    
    if env._winner == 0:
        print("\n[WIN] MCTS gagne!")
    elif env._winner == 1:
        print("\n[LOSS] Random gagne!")
    else:
        print("\n[DRAW] Match nul!")


def demo_gui(env_name: str = "tictactoe", agent_name: str = None):
    """
    Lance l'interface graphique pour OBSERVER un agent jouer.
    
    Si agent_name est donné, charge le modèle sauvegardé depuis
    results/models/<env>/<agent>.pt. Sinon utilise l'agent par défaut.
    """
    print("=" * 60)
    print(f"INTERFACE GRAPHIQUE — Observer un agent ({env_name})")
    print("=" * 60)

    try:
        from deeprl.gui.game_viewer import watch_agent
    except ImportError:
        print("\n[WARNING] pygame non installé. Installez-le avec:")
        print("   pip install pygame")
        return
    
    env, agent, fps = _build_gui_env_agent(env_name, agent_name, mode="watch")
    
    print(f"\nEnvironnement: {env.name}")
    print(f"Agent: {agent.name}")
    print("\nContrôles:")
    print("   SPACE: Pause")
    print("   N: Step-by-step")
    print("   ↑/↓: Vitesse")
    print("   ESC: Quitter")
    
    watch_agent(env, agent, n_episodes=10, fps=fps)


def demo_human_vs_agent(env_name: str = "tictactoe", agent_name: str = None):
    """
    Mode Humain vs Agent.
    
    Supporte les jeux 2 joueurs (tictactoe, quarto) ET les jeux 1 joueur
    (lineworld, gridworld) où l'humain joue directement.
    
    Si agent_name est donné, charge le modèle sauvegardé.
    """
    print("=" * 60)
    print(f"MODE: HUMAIN vs AGENT ({env_name})")
    print("=" * 60)
    
    try:
        from deeprl.gui.game_viewer import play_human_vs_agent, GameViewer
    except ImportError:
        print("\n[WARNING] pygame non installé. Installez-le avec:")
        print("   pip install pygame")
        return
    
    # Jeux 1 joueur → humain joue directement (pas d'agent adversaire)
    if env_name in ("lineworld", "gridworld"):
        if env_name == "lineworld":
            env = LineWorld(size=5)
        else:
            env = GridWorld.create_simple(size=5)
        
        print(f"\nEnvironnement: {env.name}")
        print("Mode: Humain (vous jouez!)")
        print("\nContrôles:")
        if "line" in env_name:
            print("   ←/→: Se déplacer")
        else:
            print("   ↑/↓/←/→: Se déplacer")
        print("   ESC: Quitter")
        
        viewer = GameViewer(env, agent=None, fps=30)
        viewer.run(n_episodes=5)
        return
    
    # Jeux 2 joueurs → charger / créer l'agent adversaire
    env, agent, _ = _build_gui_env_agent(env_name, agent_name, mode="play")
    
    print(f"\nEnvironnement: {env.name}")
    print(f"Adversaire: {agent.name}")
    print("\nContrôles:")
    if "quarto" in env_name:
        print("   Phase DONNER: cliquez sur une pièce ou touche 0-9/A-F")
        print("   Phase PLACER: cliquez sur une case ou touche 0-9/A-F")
    else:
        print("   Cliquez sur une case ou touches 1-9")
    print("   SPACE: Pause | ESC: Quitter")
    print("\nVous jouez en premier")
    
    n_games = 3 if "quarto" in env_name else 5
    play_human_vs_agent(env, agent, n_games=n_games, human_first=True)


def _build_gui_env_agent(env_name: str, agent_name: str = None, mode: str = "watch"):
    """
    Construit l'environnement et l'agent pour le GUI.
    
    Logique:
    1. Si agent_name fourni → utilise cet agent
    2. Sinon → utilise l'agent par défaut pour cet env
    3. Si un modèle .pt existe dans results/models/ → le charge
    4. Sinon, pour les agents apprénants → entraînement rapide
    5. Pour MCTS/RandomRollout → pas besoin de modèle
    
    Returns:
        (env, agent, fps)
    """
    # Résoudre le nom de l'env de base (mcts/alphazero/reinforce/ppo → leur env)
    env_base = env_name
    if env_name in ("mcts", "alphazero"):
        env_base = "tictactoe"
    elif env_name in ("reinforce", "ppo", "muzero", "stochastic-muzero"):
        env_base = "gridworld"
    
    # Résoudre l'agent
    if agent_name is None:
        # Anciens noms d'env = implicitement un agent
        name_map = {
            "mcts": "MCTS", "alphazero": "AlphaZero",
            "reinforce": "REINFORCE_Critic", "ppo": "PPO",
            "muzero": "MuZero", "stochastic-muzero": "MuZeroStochastic",
        }
        agent_name = name_map.get(env_name, DEFAULT_AGENT.get(env_base, "Random"))
    
    # Vérifier que l'agent existe pour cet env
    registry = AGENT_REGISTRY.get(env_base, {})
    if agent_name not in registry:
        available = list(registry.keys())
        print(f"\n[ERREUR] Agent '{agent_name}' non disponible pour '{env_base}'.")
        print(f"  Agents disponibles: {available}")
        print(f"  Usage: python main.py --gui --env {env_base} --agent {available[0] if available else '...'}")
        sys.exit(1)
    
    # Créer l'environnement
    if env_base == "lineworld":
        env = LineWorld(size=5)
    elif env_base == "gridworld":
        env = GridWorld.create_simple(size=5)
    elif env_base == "tictactoe":
        if mode == "play":
            env = TicTacToe()  # Jeu pur pour humain vs IA
        else:
            env = TicTacToe()  # Observation
    elif env_base == "quarto":
        env = Quarto()
    else:
        env = GridWorld.create_simple(size=5)
    
    # Créer l'agent
    agent = registry[agent_name]()
    
    # Tenter de charger un modèle sauvegardé
    safe_name = agent_name.replace(" ", "_").replace("/", "_")
    model_path = os.path.join("results", "models", env_base, f"{safe_name}.pt")
    
    if agent_name in NO_TRAINING_AGENTS:
        # MCTS, Random, RandomRollout → pas de modèle à charger
        print(f"\n  Agent: {agent_name} (pas de modèle nécessaire)")
    elif os.path.exists(model_path):
        # Modèle trouvé → charger
        agent.load(model_path)
        agent.set_training_mode(False)
        print(f"\n  Modèle chargé: {model_path}")
        print(f"  ({agent.episodes_played} épisodes d'entraînement)")
    else:
        # Pas de modèle → entraînement rapide
        print(f"\n  [INFO] Pas de modèle trouvé pour {agent_name} ({model_path})")
        print(f"  Entraînement rapide en cours...")
        n_ep = 2000 if env_base in ("gridworld", "lineworld") else 1000
        needs_env = agent_name in NEEDS_ENV_AGENTS
        _quick_train(env, agent, n_ep, needs_env=needs_env)
        agent.set_training_mode(False)
        print(f"  Entraînement terminé ({n_ep} épisodes)")
    
    return env, agent, 2


def _quick_train(env, agent, n_episodes: int, needs_env: bool = False):
    """Entraîne rapidement un agent (pour le GUI)."""
    trainer = Trainer(env, agent, verbose=True, log_interval=max(100, n_episodes // 5))
    trainer.train(n_episodes=n_episodes, max_steps_per_episode=100)


def demo_alphazero():
    """
    Demonstration d'AlphaZero sur TicTacToe.
    
    AlphaZero = MCTS guide par un reseau de neurones (policy + value).
    """
    print("=" * 60)
    print("DEMO: AlphaZero (Neural MCTS)")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = TicTacToe()
    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")
    
    # 2. Creer l'agent AlphaZero
    print("\n" + "=" * 60)
    print("Creation de l'agent AlphaZero")
    print("=" * 60)
    
    alphazero = AlphaZeroAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[32, 32],  # Petit reseau pour demo
        n_simulations=10,  # Tres reduit pour la demo
        c_puct=1.0,
        temperature=1.0,
        lr=1e-3
    )
    print(f"Agent: {alphazero.name}")
    print(f"Config: {alphazero.get_config()}")
    
    # 3. Self-play pour generer des donnees
    print("\n" + "=" * 60)
    print("Self-Play (generation de donnees)")
    print("=" * 60)
    
    n_self_play = 5  # Tres reduit pour la demo rapide
    all_examples = []
    
    print(f"Generation de {n_self_play} parties de self-play...")
    for i in range(n_self_play):
        examples = alphazero.self_play(env)
        all_examples.extend(examples)
        
        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/{n_self_play} parties, {len(all_examples)} positions collectees")
    
    print(f"\n[OK] {len(all_examples)} positions de training generees")
    
    # 4. Entrainement sur les donnees de self-play
    print("\n" + "=" * 60)
    print("Entrainement du reseau")
    print("=" * 60)
    
    n_epochs = 5
    batch_size = 32
    
    for epoch in range(n_epochs):
        metrics = alphazero.train_on_examples(
            all_examples, n_epochs=1, batch_size=batch_size
        )
        print(f"Epoch {epoch+1}: Policy Loss = {metrics['policy_loss']:.4f}, Value Loss = {metrics['value_loss']:.4f}")
    
    # 5. Evaluation contre Random
    print("\n" + "=" * 60)
    print("Evaluation: AlphaZero vs Random")
    print("=" * 60)
    
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    mcts_agent = MCTSAgent(n_simulations=20)  # Reduit pour la demo
    
    def play_match(env, agent1, agent2, n1, n2, n_games=20):
        wins = {n1: 0, n2: 0, "Nul": 0}
        
        for i in range(n_games):
            state = env.reset()
            agents = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]
            names = [n1, n2] if i % 2 == 0 else [n2, n1]
            
            while not env.is_game_over:
                current = env.current_player
                available = env.get_available_actions()
                
                if hasattr(agents[current], 'n_simulations'):
                    action = agents[current].act(state, available, env=env, training=False)
                else:
                    action = agents[current].act(state, available, training=False)
                
                state, reward, done = env.step(action)
            
            if env._winner == 0:
                wins[names[0]] += 1
            elif env._winner == 1:
                wins[names[1]] += 1
            else:
                wins["Nul"] += 1
        
        return wins
    
    n_games = 10  # Reduit pour la demo
    
    print(f"\n--- AlphaZero vs Random ({n_games} parties) ---")
    wins = play_match(env, alphazero, random_agent, "AlphaZero", "Random", n_games)
    print(f"   AlphaZero: {wins['AlphaZero']} ({wins['AlphaZero']/n_games*100:.0f}%)")
    print(f"   Random: {wins['Random']} ({wins['Random']/n_games*100:.0f}%)")
    print(f"   Nuls: {wins['Nul']} ({wins['Nul']/n_games*100:.0f}%)")
    
    print(f"\n--- AlphaZero vs MCTS ({n_games} parties) ---")
    wins = play_match(env, alphazero, mcts_agent, "AlphaZero", "MCTS", n_games)
    print(f"   AlphaZero: {wins['AlphaZero']} ({wins['AlphaZero']/n_games*100:.0f}%)")
    print(f"   MCTS: {wins['MCTS']} ({wins['MCTS']/n_games*100:.0f}%)")
    print(f"   Nuls: {wins['Nul']} ({wins['Nul']/n_games*100:.0f}%)")
    
    # 6. Partie de demonstration
    print("\n" + "=" * 60)
    print("PARTIE ALPHAZERO vs RANDOM")
    print("=" * 60)
    
    state = env.reset()
    print("Debut de partie:")
    env.render()
    
    while not env.is_game_over:
        available = env.get_available_actions()
        
        if env.current_player == 0:
            action = alphazero.act(state, available, env=env, training=False)
            player = "AlphaZero (X)"
        else:
            action = random_agent.act(state, available)
            player = "Random (O)"
        
        state, reward, done = env.step(action)
        print(f"\n{player} joue position {action}:")
        env.render()
    
    if env._winner == 0:
        print("\n[WIN] AlphaZero gagne!")
    elif env._winner == 1:
        print("\n[LOSS] Random gagne!")
    else:
        print("\n[DRAW] Match nul!")


def demo_muzero():
    """
    Demonstration de MuZero sur GridWorld.
    
    MuZero apprend un modele de l'environnement (dynamics + reward).
    """
    print("=" * 60)
    print("DEMO: MuZero (Learned Model)")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = GridWorld.create_simple(size=5)
    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Chemin optimal: {env.get_optimal_path_length()} steps")
    
    env.reset()
    env.render()
    
    # 2. Creer l'agent MuZero
    print("\n" + "=" * 60)
    print("Creation de l'agent MuZero")
    print("=" * 60)
    
    muzero = MuZeroAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[32],  # Tres petit reseau
        latent_dim=16,
        n_simulations=5,  # Tres reduit pour la demo
        c_puct=1.0,
        gamma=0.99,
        lr=1e-3
    )
    print(f"Agent: {muzero.name}")
    print(f"Config: {muzero.get_config()}")
    
    # 3. Collecte de donnees et entrainement
    print("\n" + "=" * 60)
    print("Entrainement MuZero")
    print("=" * 60)
    
    n_episodes = 20  # Tres reduit pour la demo
    eval_interval = 10
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            available = env.get_available_actions()
            action = muzero.act(state, available, env=env, training=True)
            next_state, reward, done = env.step(action)
            
            # Utiliser la méthode learn de MuZero
            muzero.learn(state, action, reward, next_state, done)
            state = next_state
        
        if (episode + 1) % eval_interval == 0:
            # Évaluation rapide
            eval_rewards = []
            for _ in range(5):
                s = env.reset()
                total_r = 0
                steps = 0
                while not env.is_game_over and steps < 50:
                    a = muzero.act(s, env.get_available_actions(), env=env, training=False)
                    s, r, d = env.step(a)
                    total_r += r
                    steps += 1
                eval_rewards.append(total_r)
            
            mean_reward = np.mean(eval_rewards)
            print(f"Episode {episode+1}: Reward moyen = {mean_reward:.2f}")
    
# 4. Evaluation finale
    print("\n" + "=" * 60)
    print("Evaluation finale")
    print("=" * 60)
    
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    def evaluate_agent(env, agent, n_episodes, name, use_env=False):
        rewards = []
        lengths = []
        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0
            step = 0
            while not env.is_game_over and step < 50:
                available = env.get_available_actions()
                if use_env:
                    action = agent.act(state, available, env=env, training=False)
                else:
                    action = agent.act(state, available)
                state, reward, done = env.step(action)
                total_reward += reward
                step += 1
            rewards.append(total_reward)
            lengths.append(step)
        print(f"{name}: Reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}, "
              f"Length = {np.mean(lengths):.1f} (optimal: {env.get_optimal_path_length()})")
    
    evaluate_agent(env, random_agent, 100, "Random", use_env=False)
    evaluate_agent(env, muzero, 100, "MuZero", use_env=True)
    
    # 5. Episode de demonstration
    print("\n" + "=" * 60)
    print("EPISODE AVEC MUZERO")
    print("=" * 60)
    
    state = env.reset()
    env.render()
    
    step = 0
    total_reward = 0
    while not env.is_game_over and step < 20:
        action = muzero.act(state, env.get_available_actions(), env=env, training=False)
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: {GridWorld.ACTION_NAMES[action]}, Reward = {reward:.2f}")
        env.render()
    
    print(f"\n[OK] Termine en {step} steps (optimal: {env.get_optimal_path_length()})")
    print(f"   Recompense totale: {total_reward:.2f}")


def demo_stochastic_muzero():
    """
    Demonstration de MuZero Stochastique sur GridWorld.
    
    MuZero Stochastique etend MuZero pour les environnements non-deterministes
    en modelisant explicitement les transitions stochastiques.
    
    Reference: Antonoglou et al., "Planning in Stochastic Environments 
               with a Learned Model" (2021)
    """
    print("=" * 60)
    print("DEMO: MuZero Stochastique (Stochastic Environments)")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = GridWorld.create_simple(size=5)
    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Chemin optimal: {env.get_optimal_path_length()} steps")
    
    env.reset()
    env.render()
    
    # 2. Creer l'agent MuZero Stochastique
    print("\n" + "=" * 60)
    print("Creation de l'agent MuZero Stochastique")
    print("=" * 60)
    
    stochastic_muzero = StochasticMuZeroAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dims=[32],  # Tres petit reseau
        latent_dim=16,
        chance_outcomes=8,  # Nombre de resultats stochastiques possibles
        n_simulations=5,  # Tres reduit pour la demo
        c_puct=1.0,
        gamma=0.99,
        lr=1e-3
    )
    print(f"Agent: {stochastic_muzero.name}")
    print(f"Config: {stochastic_muzero.get_config()}")
    
    # 3. Collecte de donnees et entrainement
    print("\n" + "=" * 60)
    print("Entrainement MuZero Stochastique")
    print("=" * 60)
    
    n_episodes = 20  # Tres reduit pour la demo
    eval_interval = 10
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            available = env.get_available_actions()
            action = stochastic_muzero.act(state, available, env=env, training=True)
            next_state, reward, done = env.step(action)
            
            # Utiliser la methode learn
            stochastic_muzero.learn(state, action, reward, next_state, done)
            state = next_state
        
        if (episode + 1) % eval_interval == 0:
            # Evaluation rapide
            eval_rewards = []
            for _ in range(5):
                s = env.reset()
                total_r = 0
                steps = 0
                while not env.is_game_over and steps < 50:
                    a = stochastic_muzero.act(s, env.get_available_actions(), 
                                               env=env, training=False)
                    s, r, d = env.step(a)
                    total_r += r
                    steps += 1
                eval_rewards.append(total_r)
            
            mean_reward = np.mean(eval_rewards)
            print(f"Episode {episode+1}: Reward moyen = {mean_reward:.2f}")
    
    # 4. Evaluation finale
    print("\n" + "=" * 60)
    print("Evaluation finale")
    print("=" * 60)
    
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    def evaluate_agent(env, agent, n_episodes, name, use_env=False):
        rewards = []
        lengths = []
        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0
            step = 0
            while not env.is_game_over and step < 50:
                available = env.get_available_actions()
                if use_env:
                    action = agent.act(state, available, env=env, training=False)
                else:
                    action = agent.act(state, available)
                state, reward, done = env.step(action)
                total_reward += reward
                step += 1
            rewards.append(total_reward)
            lengths.append(step)
        print(f"{name}: Reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}, "
              f"Length = {np.mean(lengths):.1f} (optimal: {env.get_optimal_path_length()})")
    
    evaluate_agent(env, random_agent, 100, "Random", use_env=False)
    evaluate_agent(env, stochastic_muzero, 100, "StochasticMuZero", use_env=True)
    
    # 5. Episode de demonstration
    print("\n" + "=" * 60)
    print("EPISODE AVEC MUZERO STOCHASTIQUE")
    print("=" * 60)
    
    state = env.reset()
    env.render()
    
    step = 0
    total_reward = 0
    while not env.is_game_over and step < 20:
        action = stochastic_muzero.act(state, env.get_available_actions(), 
                                        env=env, training=False)
        state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: {GridWorld.ACTION_NAMES[action]}, Reward = {reward:.2f}")
        env.render()
    
    print(f"\n[OK] Termine en {step} steps (optimal: {env.get_optimal_path_length()})")
    print(f"   Recompense totale: {total_reward:.2f}")


def demo_quarto():
    """
    Demonstration de Quarto avec MCTS.
    
    Quarto est un jeu strategique complexe avec 16 pieces et 10 lignes gagnantes.
    """
    print("=" * 60)
    print("DEMO: Quarto (Jeu strategique)")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = Quarto()
    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")
    print(f"   - 16 pieces avec 4 attributs binaires")
    print(f"   - 10 lignes gagnantes (4 rangees, 4 colonnes, 2 diagonales)")
    
    # 2. Montrer les pieces
    print("\nPieces disponibles (TALL/SHORT, DARK/LIGHT, SOLID/HOLLOW, SQUARE/ROUND):")
    from deeprl.envs.quarto import QuartoPiece
    pieces = QuartoPiece.all_pieces()
    for i, piece in enumerate(pieces):
        print(f"   {i:2d}: {piece}")
    
    # 3. Créer les agents
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    mcts_agent = MCTSAgent(n_simulations=100)
    
    print(f"\nAgents:")
    print(f"   - {random_agent.name}")
    print(f"   - {mcts_agent.name}")
    
    # 4. Tournoi MCTS vs Random
    print("\n" + "=" * 60)
    print("TOURNOI: MCTS vs Random")
    print("=" * 60)
    
    n_games = 10
    wins = {"MCTS": 0, "Random": 0, "Nul": 0}
    
    for i in range(n_games):
        state = env.reset()
        
        # Alterner qui commence
        agents = [mcts_agent, random_agent] if i % 2 == 0 else [random_agent, mcts_agent]
        names = ["MCTS", "Random"] if i % 2 == 0 else ["Random", "MCTS"]
        
        while not env.is_game_over:
            current = env.current_player
            available = env.get_available_actions()
            
            if hasattr(agents[current], 'n_simulations'):
                action = agents[current].act(state, available, env=env)
            else:
                action = agents[current].act(state, available)
            
            state, reward, done = env.step(action)
        
        if env._winner == 0:
            wins[names[0]] += 1
        elif env._winner == 1:
            wins[names[1]] += 1
        else:
            wins["Nul"] += 1
        
        print(f"Partie {i+1}: Gagnant = {names[env._winner] if env._winner is not None else 'Nul'}")
    
    print(f"\nResultats ({n_games} parties):")
    print(f"   MCTS: {wins['MCTS']} victoires ({wins['MCTS']/n_games*100:.0f}%)")
    print(f"   Random: {wins['Random']} victoires ({wins['Random']/n_games*100:.0f}%)")
    print(f"   Nuls: {wins['Nul']} ({wins['Nul']/n_games*100:.0f}%)")
    
    # 5. Partie de demonstration
    print("\n" + "=" * 60)
    print("PARTIE MCTS vs RANDOM")
    print("=" * 60)
    
    state = env.reset()
    print("Debut de partie:")
    env.render()
    
    step = 0
    while not env.is_game_over and step < 32:  # Max 16 pièces = 32 actions
        available = env.get_available_actions()
        
        if env.current_player == 0:
            action = mcts_agent.act(state, available, env=env)
            player = "MCTS"
        else:
            action = random_agent.act(state, available)
            player = "Random"
        
        state, reward, done = env.step(action)
        step += 1
        
        if env._phase == "place":
            print(f"\n{player} donne la pièce {action}")
        else:
            row, col = action // 4, action % 4
            print(f"\n{player} place en ({row}, {col})")
        env.render()
    
    if env._winner == 0:
        print("\n[WIN] MCTS gagne!")
    elif env._winner == 1:
        print("\n[LOSS] Random gagne!")
    else:
        print("\n[DRAW] Match nul!")


def demo_imitation():
    """
    Demonstration de l'apprentissage par imitation.
    
    Compare Behavior Cloning et DAgger.
    """
    print("=" * 60)
    print("DEMO: Expert Apprentice (Imitation Learning)")
    print("=" * 60)
    
    # 1. Creer l'environnement
    env = TicTacToe()
    print(f"\nEnvironnement: {env}")
    
    # 2. Creer l'expert (MCTS)
    expert = MCTSExpert(n_simulations=50)
    print(f"\nExpert: {expert.name}")
    
    # 3. Behavior Cloning
    print("\n" + "=" * 60)
    print("Behavior Cloning")
    print("=" * 60)
    
    bc_agent = ExpertApprenticeAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        expert=expert,
        mode="bc",
        hidden_dims=[128, 128],
        lr=1e-3
    )
    print(f"Agent: {bc_agent.name}")
    
    # Collecte de démonstrations
    print("\nCollecte de démonstrations de l'expert...")
    n_demos = 50
    
    for i in range(n_demos):
        state = env.reset()
        while not env.is_game_over:
            available = env.get_available_actions()
            
            if env.current_player == 0:
                # L'apprenti observe l'expert
                expert_action = expert.get_action(state, env=env, available_actions=available)
                bc_agent.learner.add_demonstration(state, expert_action)
                action = expert_action
            else:
                # Adversaire random
                action = np.random.choice(available)
            
            state, _, _ = env.step(action)
        
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{n_demos} demonstrations collectees, total: {len(bc_agent.learner.demonstrations)}")
    
    # Entrainement
    print("\nEntrainement sur les demonstrations...")
    metrics = bc_agent.learner.train(n_epochs=10, batch_size=32)
    print(f"   Metriques finales: loss={metrics.get('loss', 0):.4f}")
    
    # 4. DAgger
    print("\n" + "=" * 60)
    print("DAgger (Dataset Aggregation)")
    print("=" * 60)
    
    dagger_agent = ExpertApprenticeAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        expert=expert,
        mode="dagger",
        hidden_dims=[128, 128],
        lr=1e-3
    )
    print(f"Agent: {dagger_agent.name}")
    
    # Itérations DAgger (utilise collect_and_train)
    n_iterations = 3
    
    for iteration in range(n_iterations):
        metrics = dagger_agent.learner.collect_and_train(
            env=env,
            n_episodes=10,
            n_epochs_train=20,
            verbose=False
        )
        print(f"Iteration {iteration + 1}: Loss = {metrics.get('loss', 0):.4f}, "
              f"Dataset size = {metrics.get('total_demonstrations', 0)}, "
              f"Beta = {metrics.get('beta', 0):.2f}")
    
    # 5. Evaluation comparative
    print("\n" + "=" * 60)
    print("Evaluation comparative")
    print("=" * 60)
    
    random_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    
    def evaluate_vs_random(env, agent, name, n_games=50):
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(n_games):
            state = env.reset()
            
            while not env.is_game_over:
                available = env.get_available_actions()
                
                if env.current_player == 0:
                    if hasattr(agent, 'n_simulations'):
                        action = agent.act(state, available, env=env)
                    else:
                        action = agent.act(state, available, training=False)
                else:
                    action = np.random.choice(available)
                
                state, _, _ = env.step(action)
            
            if env._winner == 0:
                wins += 1
            elif env._winner == 1:
                losses += 1
            else:
                draws += 1
        
        print(f"{name}: Wins = {wins/n_games*100:.0f}%, Losses = {losses/n_games*100:.0f}%, Draws = {draws/n_games*100:.0f}%")
        return wins / n_games
    
    evaluate_vs_random(env, random_agent, "Random")
    evaluate_vs_random(env, bc_agent, "Behavior Cloning")
    evaluate_vs_random(env, dagger_agent, "DAgger")
    
    # 6. Partie de demonstration
    print("\n" + "=" * 60)
    print("PARTIE DAGGER vs RANDOM")
    print("=" * 60)
    
    state = env.reset()
    print("Debut de partie:")
    env.render()
    
    while not env.is_game_over:
        available = env.get_available_actions()
        
        if env.current_player == 0:
            action = dagger_agent.act(state, available, training=False)
            player = "DAgger (X)"
        else:
            action = np.random.choice(available)
            player = "Random (O)"
        
        state, reward, done = env.step(action)
        print(f"\n{player} joue position {action}:")
        env.render()
    
    if env._winner == 0:
        print("\n[WIN] DAgger gagne!")
    elif env._winner == 1:
        print("\n[LOSS] Random gagne!")
    else:
        print("\n[DRAW] Match nul!")

def main():
    """Point d'entree principal."""
    parser = argparse.ArgumentParser(
        description="DeepRL - Bibliotheque de Deep Reinforcement Learning"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Lancer le benchmark complet"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Lancer l'interface graphique (observer un agent)"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Jouer contre un agent IA (mode humain vs agent)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent à utiliser pour --gui/--play (ex: MCTS, DQN, PPO, TabularQLearning...)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="gridworld",
        choices=["lineworld", "gridworld", "tictactoe", "reinforce", "ppo", 
                 "mcts", "alphazero", "muzero", "stochastic-muzero", "quarto", "imitation"],
        help="Environnement/algorithme a demontrer"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("    DeepRL - Deep Reinforcement Learning Library")
    print("=" * 60 + "\n")
    
    if args.benchmark:
        run_benchmark()
    elif args.gui:
        demo_gui(args.env, agent_name=args.agent)
    elif args.play:
        demo_human_vs_agent(args.env, agent_name=args.agent)
    else:
        if args.env == "lineworld":
            demo_lineworld()
        elif args.env == "gridworld":
            demo_gridworld()
        elif args.env == "tictactoe":
            demo_tictactoe()
        elif args.env == "reinforce":
            demo_reinforce()
        elif args.env == "ppo":
            demo_ppo()
        elif args.env == "mcts":
            demo_mcts()
        elif args.env == "alphazero":
            demo_alphazero()
        elif args.env == "muzero":
            demo_muzero()
        elif args.env == "stochastic-muzero":
            demo_stochastic_muzero()
        elif args.env == "quarto":
            demo_quarto()
        elif args.env == "imitation":
            demo_imitation()
    
    print("\n[OK] Termine!")


if __name__ == "__main__":
    main()