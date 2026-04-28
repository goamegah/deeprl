"""
DeepRL - Point d'entree principal

Flags:
    --env       lineworld | gridworld | tictactoe | quarto  (defaut: gridworld)
    --agent     nom de l'agent (ex: DDQN_ER, Random, Human)
                Si absent → demo texte dans le terminal
    --versus    adversaire pour jeux 2 joueurs (defaut: Random)
                N'a de sens que si --agent est fourni et env 2 joueurs
    --run       dossier de run a utiliser (ex: results/2026-04-24_11-25-20)
                Si absent → cherche dans results/latest puis tous les runs

Exemples:
    python main.py --env tictactoe                                           # stats texte
    python main.py --env tictactoe --agent DDQN_ER                           # GUI: DDQN_ER vs Random
    python main.py --env tictactoe --agent DDQN_ER --versus DDQN_PER         # GUI: agent vs agent
    python main.py --env tictactoe --agent Human --versus DDQN_ER            # GUI: vous vs DDQN_ER
    python main.py --env tictactoe --agent Human --versus Human               # GUI: pvp
    python main.py --env lineworld --agent Human                             # GUI: vous jouez
    python main.py --env lineworld --agent DDQN_ER                           # GUI: observer DDQN_ER
    python main.py --env tictactoe --agent DDQN_ER --run results/2026-04-24  # run specifique
"""

import argparse
import sys

from deeprl.envs import LineWorld, TicTacToe, Quarto
from deeprl.agents import RandomAgent
from deeprl.training import Trainer
from deeprl.registry import (
    AGENT_REGISTRY, NO_TRAINING_AGENTS,
    QUICK_TRAIN_EPISODES, DEFAULT_FPS,
    make_env, make_env_2player, find_latest_model,
)

# Environnements 2 joueurs (supportent --versus)
TWO_PLAYER_ENVS = {"tictactoe", "quarto"}


def demo_lineworld():
    """Simulation random sur LineWorld + parties/sec."""
    import time

    print("=" * 60)
    print("DEMO: LineWorld (Random)")
    print("=" * 60)

    env = LineWorld(size=5)
    agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)

    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")

    # 1. Simulation random
    n_games = 10000
    total_reward = 0

    start = time.time()
    for _ in range(n_games):
        state = env.reset()
        ep_reward = 0
        step = 0
        while not env.is_game_over and step < 200:
            action = agent.act(state, env.get_available_actions())
            state, reward, done = env.step(action)
            ep_reward += reward
            step += 1
        total_reward += ep_reward
    elapsed = time.time() - start

    print(f"\nResultats ({n_games} parties):")
    print(f"   Recompense moyenne: {total_reward / n_games:.3f}")
    print(f"   Vitesse: {n_games / elapsed:.1f} parties/sec")


def demo_gridworld():
    """Simulation random sur GridWorld + parties/sec."""
    import time

    print("=" * 60)
    print("DEMO: GridWorld (Random)")
    print("=" * 60)

    env = make_env("gridworld")
    agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)

    print(f"\nEnvironnement: {env}")
    print(f"   - State dim: {env.state_dim}")
    print(f"   - Nombre d'actions: {env.n_actions}")

    # 1. Simulation random
    n_games = 10000
    total_reward = 0

    start = time.time()
    for _ in range(n_games):
        state = env.reset()
        ep_reward = 0
        step = 0
        while not env.is_game_over and step < 200:
            action = agent.act(state, env.get_available_actions())
            state, reward, done = env.step(action)
            ep_reward += reward
            step += 1
        total_reward += ep_reward
    elapsed = time.time() - start

    print(f"\nResultats ({n_games} parties):")
    print(f"   Recompense moyenne: {total_reward / n_games:.3f}")
    print(f"   Vitesse: {n_games / elapsed:.1f} parties/sec")


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
    n_games = 10000
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

    # 2. Creer les agents
    agent1 = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    agent2 = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)

    print(f"\nJoueur 0: {agent1.name}")
    print(f"Joueur 1: {agent2.name}")

    # 3. Tournoi Random vs Random
    print("\n" + "=" * 60)
    print("TOURNOI: Random vs Random")
    print("=" * 60)

    n_games = 10000
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


def _resolve_agent(env_name: str, agent_name: str, run_dir: str = None):
    """
    Cree, charge (ou entraine rapidement) un agent pour le GUI.

    Args:
        run_dir: si fourni, cherche le modele dans run_dir/<env>/models/
                 sinon utilise find_latest_model (results/latest puis fallback).

    Returns:
        Instance d'agent prete a jouer (mode inference).
    """
    import glob, re, os as _os

    registry = AGENT_REGISTRY.get(env_name, {})
    if agent_name not in registry:
        available = list(registry.keys())
        print(f"\n[ERREUR] Agent '{agent_name}' non disponible pour '{env_name}'.")
        print(f"  Agents disponibles: {available}")
        sys.exit(1)

    agent = registry[agent_name]()

    if agent_name in NO_TRAINING_AGENTS:
        print(f"  Agent: {agent_name} (pas d'entrainement necessaire)")
    else:
        if run_dir:
            safe = agent_name.replace(" ", "_").replace("/", "_")
            pattern = _os.path.join(run_dir, env_name, "models", f"{safe}_ckpt*.pt")
            matches = sorted(glob.glob(pattern),
                             key=lambda p: int(re.search(r"_ckpt(\d+)\.pt$", p).group(1)))
            model_path = matches[-1] if matches else None
            if not model_path:
                print(f"  [ERREUR] Aucun modele pour {agent_name} dans {run_dir}/{env_name}/models/")
                sys.exit(1)
        else:
            model_path = find_latest_model(env_name, agent_name)
        if model_path:
            agent.load(model_path)
            agent.set_training_mode(False)
            print(f"  Modele charge: {model_path}")
            print(f"  ({agent.episodes_played} episodes d'entrainement)")
        else:
            n_ep = QUICK_TRAIN_EPISODES.get(env_name, 1000)
            print(f"  [INFO] Pas de modele trouve pour {agent_name}")
            print(f"  Entrainement rapide ({n_ep} episodes)...")
            train_env = make_env(env_name)
            trainer = Trainer(
                train_env, agent, verbose=True,
                log_interval=max(100, n_ep // 5)
            )
            trainer.train(n_episodes=n_ep, max_steps_per_episode=100)
            agent.set_training_mode(False)
            print("  Entrainement termine")

    return agent


def main():
    """Point d'entree principal."""
    parser = argparse.ArgumentParser(
        description="DeepRL - Deep Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py --env tictactoe                              # stats texte
  python main.py --env tictactoe --agent DDQN_ER              # GUI: DDQN_ER vs Random
  python main.py --env tictactoe --agent DDQN_ER --versus DDQN_PER  # agent vs agent
  python main.py --env tictactoe --agent Human --versus DDQN_ER     # vous vs DDQN_ER
  python main.py --env tictactoe --agent Human --versus Human        # pvp
  python main.py --env lineworld --agent Human                # vous jouez
  python main.py --env lineworld --agent DDQN_ER              # observer DDQN_ER
        """
    )
    parser.add_argument(
        "--env",
        type=str,
        default="gridworld",
        choices=list(AGENT_REGISTRY.keys()),
        help="Environnement (defaut: gridworld)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent principal — si absent: demo texte. 'Human' = vous jouez."
    )
    parser.add_argument(
        "--versus",
        type=str,
        default="Random",
        help="Adversaire pour jeux 2 joueurs (defaut: Random). 'Human' = pvp."
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="DIR",
        help="Dossier de run a utiliser pour charger les modeles "
             "(ex: results/2026-04-24_11-25-20). "
             "Si absent: cherche dans results/latest puis tous les runs."
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("    DeepRL - Deep Reinforcement Learning Library")
    print("=" * 60 + "\n")

    env_name = args.env
    agent_name = args.agent
    versus_name = args.versus
    run_dir = args.run

    # ── Mode demo texte (aucun --agent) ──────────────────────────────────────
    if agent_name is None:
        if env_name == "lineworld":
            demo_lineworld()
        elif env_name == "gridworld":
            demo_gridworld()
        elif env_name == "tictactoe":
            demo_tictactoe()
        elif env_name == "quarto":
            demo_quarto()
        print("\n[OK] Termine!")
        return

    # ── Mode GUI ──────────────────────────────────────────────────────────────
    try:
        from deeprl.gui.game_viewer import (
            GameViewer, watch_agent_vs_agent, play_human_vs_agent,
        )
    except ImportError:
        print("\n[WARNING] pygame non installe. Installez-le avec:")
        print("   pip install pygame")
        sys.exit(1)

    fps = DEFAULT_FPS.get(env_name, 2)
    is_2player = env_name in TWO_PLAYER_ENVS

    print(f"Environnement : {env_name}")
    print(f"Agent         : {agent_name}")
    if is_2player:
        print(f"Versus        : {versus_name}")
    print()

    # ── Jeux 1 joueur (lineworld, gridworld) ─────────────────────────────────
    if not is_2player:
        env = make_env_2player(env_name)
        if agent_name == "Human":
            viewer = GameViewer(env, agent=None, fps=fps,
                                title=f"Humain — {env_name}")
        else:
            agent = _resolve_agent(env_name, agent_name, run_dir)
            viewer = GameViewer(env, agent=agent, fps=fps,
                                title=f"{agent_name} — {env_name}")
        viewer.run(n_episodes=10)

    # ── Jeux 2 joueurs (tictactoe, quarto) ───────────────────────────────────
    else:
        env = make_env_2player(env_name)
        a0_human = (agent_name == "Human")
        a1_human = (versus_name == "Human")

        if a0_human and a1_human:
            # Humain vs Humain
            print("Mode: Humain vs Humain")
            print("Controles: clic souris | R: restart | ESC: quitter")
            viewer = GameViewer(env, agent=None, fps=30,
                                title="Humain vs Humain")
            viewer.run(n_episodes=10)

        elif a0_human:
            # Vous (J0) vs agent (J1)
            opponent = _resolve_agent(env_name, versus_name, run_dir)
            print(f"Mode: Vous (J0) vs {versus_name} (J1)")
            print("Controles: clic souris | SPACE: pause | ESC: quitter")
            play_human_vs_agent(env, opponent, n_games=5, human_first=True)

        elif a1_human:
            # Agent (J0) vs Vous (J1)
            agent = _resolve_agent(env_name, agent_name, run_dir)
            print(f"Mode: {agent_name} (J0) vs Vous (J1)")
            print("Controles: clic souris | SPACE: pause | ESC: quitter")
            play_human_vs_agent(env, agent, n_games=5, human_first=False)

        else:
            # Agent vs Agent
            agent_0 = _resolve_agent(env_name, agent_name, run_dir)
            agent_1 = _resolve_agent(env_name, versus_name, run_dir)
            print(f"Mode: {agent_name} (J0) vs {versus_name} (J1)")
            print("Controles: SPACE: pause | N: step | +/-: vitesse | ESC: quitter")
            watch_agent_vs_agent(env, agent_0, agent_1, n_episodes=10, fps=fps)

    print("\n[OK] Termine!")


if __name__ == "__main__":
    main()

