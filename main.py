"""
DeepRL - Point d'entree principal

Ce fichier permet de:
1. Tester rapidement les environnements et agents
2. Jouer contre un agent IA (interface graphique)
3. Observer un agent jouer (interface graphique)
4. Demontrer l'utilisation de la bibliotheque

Pour l'entrainement et les benchmarks, utiliser: python run_experiments.py

Usage:
    python main.py                    # Test rapide (GridWorld + Q-Learning)
    python main.py --gui              # Observer un agent jouer
    python main.py --play             # Jouer contre un agent IA
    python main.py --env lineworld    # Environnement specifique
    python main.py --env gridworld    # GridWorld avec Q-Learning
    python main.py --env tictactoe    # TicTacToe avec Random
    python main.py --env quarto       # Quarto avec Random
"""

import argparse
import os
import sys
import numpy as np

from deeprl.envs import LineWorld, GridWorld, TicTacToe, TicTacToeVsRandom, Quarto, QuartoVsRandom
from deeprl.agents import (
    RandomAgent, TabularQLearning
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
        "Random":                   lambda: RandomAgent(state_dim=114, n_actions=32)
    },
}

# Agents qui n'ont pas besoin d'entraînement (pas de .pt à charger)
NO_TRAINING_AGENTS = {"Random"}

# Agents qui ont besoin du param env= dans act()
NEEDS_ENV_AGENTS = set()

# Agent par défaut pour chaque env (utilisé si --agent non spécifié)
DEFAULT_AGENT = {
    "lineworld": "TabularQLearning",
    "gridworld": "TabularQLearning",
    "tictactoe": "Random",
    "quarto": "Random",
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
    if 'games_per_second' in eval_summary:
        print(f"   - Vitesse: {eval_summary['games_per_second']:.1f} parties/sec")


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
    if 'games_per_second' in summary_random:
        print(f"{'Parties/sec':<25} {summary_random['games_per_second']:.1f}                  {summary_q['games_per_second']:.1f}")
    
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
    Demonstration de TicTacToe avec des agents aleatoires.

    Joue des parties Random vs Random et affiche les statistiques.
    """
    import time

    print("=" * 60)
    print("DEMO: TicTacToe (Random vs Random)")
    print("=" * 60)

    # 1. Creer l'environnement
    env = TicTacToe()
    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")

    # 2. Creer les agents
    agent_x = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    agent_o = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)

    print(f"\nJoueur X: {agent_x.name}")
    print(f"Joueur O: {agent_o.name}")

    # 3. Jouer des parties
    n_games = 1000
    wins_x, wins_o, draws = 0, 0, 0

    start = time.time()

    for _ in range(n_games):
        state = env.reset()

        while not env.is_game_over:
            available = env.get_available_actions()
            if env.current_player == 0:
                action = agent_x.act(state, available)
            else:
                action = agent_o.act(state, available)
            state, reward, done = env.step(action)

        if env._winner == 0:
            wins_x += 1
        elif env._winner == 1:
            wins_o += 1
        else:
            draws += 1

    elapsed = time.time() - start

    print(f"\nResultats ({n_games} parties):")
    print(f"   X (Random): {wins_x} victoires ({wins_x/n_games*100:.1f}%)")
    print(f"   O (Random): {wins_o} victoires ({wins_o/n_games*100:.1f}%)")
    print(f"   Nuls: {draws} ({draws/n_games*100:.1f}%)")
    print(f"   Vitesse: {n_games/elapsed:.1f} parties/sec")

    # 4. Partie de demonstration
    print("\n" + "=" * 60)
    print("PARTIE DE DEMONSTRATION")
    print("=" * 60)

    state = env.reset()
    print("Debut de partie:")
    env.render()

    while not env.is_game_over:
        available = env.get_available_actions()
        if env.current_player == 0:
            action = agent_x.act(state, available)
            player = "X"
        else:
            action = agent_o.act(state, available)
            player = "O"

        state, reward, done = env.step(action)
        print(f"\n{player} joue position {action}:")
        env.render()

    if env._winner == 0:
        print("\nX gagne!")
    elif env._winner == 1:
        print("\nO gagne!")
    else:
        print("\nMatch nul!")

    # 5. Evaluation avec TicTacToeVsRandom
    print("\n" + "=" * 60)
    print("EVALUATION: Random vs RandomBot")
    print("=" * 60)

    env_vs = TicTacToeVsRandom()
    agent = RandomAgent(state_dim=env_vs.state_dim, n_actions=env_vs.n_actions)

    evaluator = Evaluator(env_vs, agent, verbose=True)
    results = evaluator.evaluate(n_episodes=200)
    summary = results.get_summary()

    print(f"\nScore moyen: {summary['mean_score']:.3f} +/- {summary['std_score']:.3f}")
    if 'win_rate' in summary:
        print(f"Win: {summary['win_rate']:.1%}, Loss: {summary['loss_rate']:.1%}, Draw: {summary['draw_rate']:.1%}")


def demo_quarto():
    """
    Demonstration de Quarto avec des agents aleatoires.

    Quarto est un jeu strategique complexe avec 16 pieces et 10 lignes gagnantes.
    """
    import time

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

    # 3. Creer les agents
    agent1 = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    agent2 = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)

    print(f"\nJoueur 0: {agent1.name}")
    print(f"Joueur 1: {agent2.name}")

    # 4. Tournoi Random vs Random
    print("\n" + "=" * 60)
    print("TOURNOI: Random vs Random")
    print("=" * 60)

    n_games = 100
    wins = {"J0": 0, "J1": 0, "Nul": 0}

    start = time.time()

    for i in range(n_games):
        state = env.reset()
        agents = [agent1, agent2]

        while not env.is_game_over:
            current = env.current_player
            available = env.get_available_actions()
            action = agents[current].act(state, available)
            state, reward, done = env.step(action)

        if env._winner == 0:
            wins["J0"] += 1
        elif env._winner == 1:
            wins["J1"] += 1
        else:
            wins["Nul"] += 1

    elapsed = time.time() - start

    print(f"\nResultats ({n_games} parties):")
    print(f"   Joueur 0: {wins['J0']} victoires ({wins['J0']/n_games*100:.0f}%)")
    print(f"   Joueur 1: {wins['J1']} victoires ({wins['J1']/n_games*100:.0f}%)")
    print(f"   Nuls: {wins['Nul']} ({wins['Nul']/n_games*100:.0f}%)")
    print(f"   Vitesse: {n_games/elapsed:.1f} parties/sec")

    # 5. Evaluation avec QuartoVsRandom
    print("\n" + "=" * 60)
    print("EVALUATION: Random vs RandomBot")
    print("=" * 60)

    env_vs = QuartoVsRandom()
    agent = RandomAgent(state_dim=env_vs.state_dim, n_actions=env_vs.n_actions)

    evaluator = Evaluator(env_vs, agent, verbose=True)
    results = evaluator.evaluate(n_episodes=200)
    summary = results.get_summary()

    print(f"\nScore moyen: {summary['mean_score']:.3f} +/- {summary['std_score']:.3f}")
    if 'win_rate' in summary:
        print(f"Win: {summary['win_rate']:.1%}, Loss: {summary['loss_rate']:.1%}, Draw: {summary['draw_rate']:.1%}")
    if 'games_per_second' in summary:
        print(f"Vitesse: {summary['games_per_second']:.1f} parties/sec")

    # 6. Partie de demonstration
    print("\n" + "=" * 60)
    print("PARTIE DE DEMONSTRATION")
    print("=" * 60)

    state = env.reset()
    print("Debut de partie:")
    env.render()

    step = 0
    while not env.is_game_over and step < 32:
        available = env.get_available_actions()

        if env.current_player == 0:
            action = agent1.act(state, available)
            player = "Joueur 0"
        else:
            action = agent2.act(state, available)
            player = "Joueur 1"

        state, reward, done = env.step(action)
        step += 1

        if env._phase == "place":
            print(f"\n{player} donne la piece {action}")
        else:
            row, col = action // 4, action % 4
            print(f"\n{player} place en ({row}, {col})")
        env.render()

    if env._winner == 0:
        print("\nJoueur 0 gagne!")
    elif env._winner == 1:
        print("\nJoueur 1 gagne!")
    else:
        print("\nMatch nul!")


def demo_gui(env_name: str = "tictactoe", agent_name: str = None):
    """
    Lance l'interface graphique pour OBSERVER un agent jouer.
    """
    print("=" * 60)
    print(f"INTERFACE GRAPHIQUE — Observer un agent ({env_name})")
    print("=" * 60)

    try:
        from deeprl.gui.game_viewer import watch_agent
    except ImportError:
        print("\n[WARNING] pygame non installe. Installez-le avec:")
        print("   pip install pygame")
        return

    env, agent, fps = _build_gui_env_agent(env_name, agent_name, mode="watch")

    print(f"\nEnvironnement: {env.name}")
    print(f"Agent: {agent.name}")
    print("\nControles:")
    print("   SPACE: Pause")
    print("   N: Step-by-step")
    print("   UP/DOWN: Vitesse")
    print("   ESC: Quitter")

    watch_agent(env, agent, n_episodes=10, fps=fps)


def demo_pvp(env_name: str = "tictactoe"):
    """
    Mode Humain vs Humain (2 joueurs sur le meme ecran).

    Chaque joueur joue a tour de role avec la souris/clavier.
    Supporte les jeux 2 joueurs: tictactoe et quarto.
    """
    print("=" * 60)
    print(f"MODE: HUMAIN vs HUMAIN ({env_name})")
    print("=" * 60)

    try:
        from deeprl.gui.game_viewer import GameViewer
    except ImportError:
        print("\n[WARNING] pygame non installe. Installez-le avec:")
        print("   pip install pygame")
        return

    if env_name not in ("tictactoe", "quarto"):
        print(f"\n[INFO] Le mode PvP n'est disponible que pour les jeux 2 joueurs.")
        print("   Environnements supportes: tictactoe, quarto")
        return

    if env_name == "tictactoe":
        env = TicTacToe()
    else:
        env = Quarto()

    print(f"\nEnvironnement: {env.name}")
    print("Mode: Humain vs Humain (meme ecran)")
    print("\nControles:")
    if "quarto" in env_name:
        print("   Phase DONNER: cliquez sur une piece dans le panel droit")
        print("   Phase PLACER: cliquez sur une case du plateau")
    else:
        print("   Cliquez sur une case ou touches 1-9")
    print("   [R] Restart  [SPACE] Pause  [ESC] Quitter")
    print("\nJoueur 0 commence!")

    viewer = GameViewer(env, agent=None, fps=30, title="Humain vs Humain")
    viewer.run(n_episodes=10)


def demo_human_vs_agent(env_name: str = "tictactoe", agent_name: str = None):
    """
    Mode Humain vs Agent.

    Supporte les jeux 2 joueurs (tictactoe, quarto) ET les jeux 1 joueur
    (lineworld, gridworld) ou l'humain joue directement.
    """
    print("=" * 60)
    print(f"MODE: HUMAIN vs AGENT ({env_name})")
    print("=" * 60)

    try:
        from deeprl.gui.game_viewer import play_human_vs_agent, GameViewer
    except ImportError:
        print("\n[WARNING] pygame non installe. Installez-le avec:")
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
        print("\nControles:")
        if "line" in env_name:
            print("   LEFT/RIGHT: Se deplacer")
        else:
            print("   UP/DOWN/LEFT/RIGHT: Se deplacer")
        print("   ESC: Quitter")

        viewer = GameViewer(env, agent=None, fps=30)
        viewer.run(n_episodes=5)
        return

    # Jeux 2 joueurs → charger / creer l'agent adversaire
    env, agent, _ = _build_gui_env_agent(env_name, agent_name, mode="play")

    print(f"\nEnvironnement: {env.name}")
    print(f"Adversaire: {agent.name}")
    print("\nControles:")
    if "quarto" in env_name:
        print("   Phase DONNER: cliquez sur une piece ou touche 0-9/A-F")
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
    2. Sinon → utilise l'agent par defaut pour cet env
    3. Si un modele .pt existe dans results/models/ → le charge
    4. Sinon, pour les agents apprenants → entrainement rapide

    Returns:
        (env, agent, fps)
    """
    env_base = env_name

    # Resoudre l'agent
    if agent_name is None:
        agent_name = DEFAULT_AGENT.get(env_base, "Random")

    # Verifier que l'agent existe pour cet env
    registry = AGENT_REGISTRY.get(env_base, {})
    if agent_name not in registry:
        available = list(registry.keys())
        print(f"\n[ERREUR] Agent '{agent_name}' non disponible pour '{env_base}'.")
        print(f"  Agents disponibles: {available}")
        print(f"  Usage: python main.py --gui --env {env_base} --agent {available[0] if available else '...'}")
        sys.exit(1)

    # Creer l'environnement
    if env_base == "lineworld":
        env = LineWorld(size=5)
    elif env_base == "gridworld":
        env = GridWorld.create_simple(size=5)
    elif env_base == "tictactoe":
        env = TicTacToe()
    elif env_base == "quarto":
        env = Quarto()
    else:
        env = GridWorld.create_simple(size=5)

    # Creer l'agent
    agent = registry[agent_name]()

    # Tenter de charger un modele sauvegarde
    safe_name = agent_name.replace(" ", "_").replace("/", "_")
    model_path = os.path.join("results", "models", env_base, f"{safe_name}.pt")

    if agent_name in NO_TRAINING_AGENTS:
        # Random → pas de modele a charger
        print(f"\n  Agent: {agent_name} (pas de modele necessaire)")
    elif os.path.exists(model_path):
        # Modele trouve → charger
        agent.load(model_path)
        agent.set_training_mode(False)
        print(f"\n  Modele charge: {model_path}")
        print(f"  ({agent.episodes_played} episodes d'entrainement)")
    else:
        # Pas de modele → entrainement rapide
        print(f"\n  [INFO] Pas de modele trouve pour {agent_name} ({model_path})")
        print(f"  Entrainement rapide en cours...")
        n_ep = 2000 if env_base in ("gridworld", "lineworld") else 1000
        _quick_train(env, agent, n_ep)
        agent.set_training_mode(False)
        print(f"  Entrainement termine ({n_ep} episodes)")

    return env, agent, 2


def _quick_train(env, agent, n_episodes: int):
    """Entraine rapidement un agent (pour le GUI)."""
    trainer = Trainer(env, agent, verbose=True, log_interval=max(100, n_episodes // 5))
    trainer.train(n_episodes=n_episodes, max_steps_per_episode=100)


def main():
    """Point d'entree principal."""
    parser = argparse.ArgumentParser(
        description="DeepRL - Bibliotheque de Deep Reinforcement Learning"
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
        "--pvp",
        action="store_true",
        help="Mode 2 joueurs humains sur le meme ecran (tictactoe, quarto)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent a utiliser pour --gui/--play (ex: TabularQLearning, Random)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="gridworld",
        choices=["lineworld", "gridworld", "tictactoe", "quarto"],
        help="Environnement a demontrer"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("    DeepRL - Deep Reinforcement Learning Library")
    print("=" * 60 + "\n")

    if args.gui:
        demo_gui(args.env, agent_name=args.agent)
    elif args.play:
        demo_human_vs_agent(args.env, agent_name=args.agent)
    elif args.pvp:
        demo_pvp(args.env)
    else:
        if args.env == "lineworld":
            demo_lineworld()
        elif args.env == "gridworld":
            demo_gridworld()
        elif args.env == "tictactoe":
            demo_tictactoe()
        elif args.env == "quarto":
            demo_quarto()

    print("\n[OK] Termine!")


if __name__ == "__main__":
    main()
