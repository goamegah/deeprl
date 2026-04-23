"""
Registre centralise des agents et des environnements.

Source unique de verite pour :
- AGENT_REGISTRY : (env, agent_name) -> lambda qui cree l'agent
- make_env / make_env_2player : factory d'environnements
- NO_TRAINING_AGENTS : agents sans apprentissage
- DEFAULT_AGENT : agent par defaut pour chaque env (GUI)

Importe par main.py et run_experiments.py pour eviter la duplication.
"""

from deeprl.envs import (
    LineWorld, GridWorld,
    TicTacToe, TicTacToeVsRandom,
    Quarto, QuartoVsRandom,
)
from deeprl.agents import (
    RandomAgent, TabularQLearning,
    DeepQLearning, DoubleDeepQLearning,
    DDQNWithExperienceReplay, DDQNWithPrioritizedExperienceReplay,
    REINFORCE, REINFORCEWithMeanBaseline, REINFORCEWithCriticBaseline, PPO,
    RandomRollout, MCTS, AlphaZero, MuZero, MuZeroStochastic,
    ExpertApprentice,
)


# ============================================================================
# REGISTRE DES AGENTS — hyperparametres calibres par environnement
# ============================================================================

AGENT_REGISTRY = {
    # -----------------------------------------------------------------------
    # LINEWORLD — 5 états, 2 actions (gauche / droite)
    # Épisodes courts (~2 steps) → epsilon_decay rapide, buffer petit
    # -----------------------------------------------------------------------
    "lineworld": {
        "Random": lambda: RandomAgent(
            state_dim=5, n_actions=2,
        ),
        "TabularQLearning": lambda: TabularQLearning(
            n_states=5, n_actions=2,
            lr=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        ),
        "DeepQLearning": lambda: DeepQLearning(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        ),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=50,
        ),
        "DDQN_ER": lambda: DDQNWithExperienceReplay(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=50,
            buffer_size=5_000, batch_size=32, min_buffer_size=500,
        ),
        "DDQN_PER": lambda: DDQNWithPrioritizedExperienceReplay(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=50,
            buffer_size=5_000, batch_size=32, min_buffer_size=500,
            alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=500_000,
        ),
        "REINFORCE": lambda: REINFORCE(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99,
        ),
        "REINFORCEWithMeanBaseline": lambda: REINFORCEWithMeanBaseline(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99,
        ),
        "REINFORCEWithCriticBaseline": lambda: REINFORCEWithCriticBaseline(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, lr_critic=1e-3, gamma=0.99,
        ),
        "PPO": lambda: PPO(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, lr_critic=1e-3, gamma=0.99,
            clip_eps=0.2, n_epochs=4, entropy_coef=0.01, value_coef=0.5,
        ),
        "RandomRollout": lambda: RandomRollout(
            state_dim=5, n_actions=2, n_simulations=20, max_depth=50, gamma=1.0,
        ),
        "MCTS": lambda: MCTS(
            state_dim=5, n_actions=2, n_simulations=50, c_puct=1.41, max_depth=50, gamma=1.0,
        ),
        "AlphaZero": lambda: AlphaZero(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, gamma=0.99, n_simulations=25, c_puct=1.0,
            temperature=1.0, l2_reg=1e-4,
            buffer_size=5_000, batch_size=32, min_buffer_size=200,
        ),
        "MuZero": lambda: MuZero(
            state_dim=5, n_actions=2, latent_dim=16, hidden_dims=[32],
            lr=1e-3, gamma=0.99, n_simulations=25, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=5_000, batch_size=32, min_buffer_size=200,
        ),
        "MuZeroStochastic": lambda: MuZeroStochastic(
            state_dim=5, n_actions=2, latent_dim=16, n_chance=4, hidden_dims=[32],
            lr=1e-3, gamma=0.99, n_simulations=25, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=5_000, batch_size=32, min_buffer_size=200,
        ),
        "ExpertApprentice": lambda: ExpertApprentice(
            state_dim=5, n_actions=2, hidden_dims=[32],
            lr=1e-3, n_simulations=30, c_puct=1.41, gamma=1.0,
            max_depth=50, l2_reg=1e-4, use_student_ratio=0.0,
            buffer_size=5_000, batch_size=32, min_buffer_size=200,
        ),
    },
    # -----------------------------------------------------------------------
    # GRIDWORLD — 25 états, 4 actions, grille 5×5
    # Épisodes moyens (~20 steps) → buffers médiums
    # -----------------------------------------------------------------------
    "gridworld": {
        "Random": lambda: RandomAgent(
            state_dim=25, n_actions=4,
        ),
        "TabularQLearning": lambda: TabularQLearning(
            n_states=25, n_actions=4,
            lr=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        ),
        "DeepQLearning": lambda: DeepQLearning(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        ),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=100,
        ),
        "DDQN_ER": lambda: DDQNWithExperienceReplay(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=100,
            buffer_size=10_000, batch_size=32, min_buffer_size=1_000,
        ),
        "DDQN_PER": lambda: DDQNWithPrioritizedExperienceReplay(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=1e-3, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=100,
            buffer_size=10_000, batch_size=32, min_buffer_size=1_000,
            alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=2_000_000,
        ),
        "REINFORCE": lambda: REINFORCE(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=5e-4, gamma=0.99,
        ),
        "REINFORCEWithMeanBaseline": lambda: REINFORCEWithMeanBaseline(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=5e-4, gamma=0.99,
        ),
        "REINFORCEWithCriticBaseline": lambda: REINFORCEWithCriticBaseline(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=5e-4, lr_critic=5e-4, gamma=0.99,
        ),
        "PPO": lambda: PPO(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=5e-4, lr_critic=5e-4, gamma=0.99,
            clip_eps=0.2, n_epochs=4, entropy_coef=0.01, value_coef=0.5,
        ),
        "RandomRollout": lambda: RandomRollout(
            state_dim=25, n_actions=4, n_simulations=20, max_depth=100, gamma=1.0,
        ),
        "MCTS": lambda: MCTS(
            state_dim=25, n_actions=4, n_simulations=100, c_puct=1.41, max_depth=100, gamma=1.0,
        ),
        "AlphaZero": lambda: AlphaZero(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=5e-4, gamma=0.99, n_simulations=50, c_puct=1.0,
            temperature=1.0, l2_reg=1e-4,
            buffer_size=10_000, batch_size=32, min_buffer_size=500,
        ),
        "MuZero": lambda: MuZero(
            state_dim=25, n_actions=4, latent_dim=32, hidden_dims=[64, 32],
            lr=5e-4, gamma=0.99, n_simulations=50, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=10_000, batch_size=32, min_buffer_size=500,
        ),
        "MuZeroStochastic": lambda: MuZeroStochastic(
            state_dim=25, n_actions=4, latent_dim=32, n_chance=4, hidden_dims=[64, 32],
            lr=5e-4, gamma=0.99, n_simulations=50, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=10_000, batch_size=32, min_buffer_size=500,
        ),
        "ExpertApprentice": lambda: ExpertApprentice(
            state_dim=25, n_actions=4, hidden_dims=[64, 32],
            lr=5e-4, n_simulations=50, c_puct=1.41, gamma=1.0,
            max_depth=100, l2_reg=1e-4, use_student_ratio=0.0,
            buffer_size=10_000, batch_size=32, min_buffer_size=500,
        ),
    },
    # -----------------------------------------------------------------------
    # TICTACTOE — 27 dims, 9 actions, épisodes ~5-9 steps
    # epsilon_decay lent (0.9995) → exploration sur ~9K épisodes
    # -----------------------------------------------------------------------
    "tictactoe": {
        "Random": lambda: RandomAgent(
            state_dim=27, n_actions=9,
        ),
        "DeepQLearning": lambda: DeepQLearning(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=5e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
        ),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=5e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
            target_update_freq=200,
        ),
        "DDQN_ER": lambda: DDQNWithExperienceReplay(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=5e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
            target_update_freq=200,
            buffer_size=20_000, batch_size=32, min_buffer_size=1_000,
        ),
        "DDQN_PER": lambda: DDQNWithPrioritizedExperienceReplay(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=5e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
            target_update_freq=200,
            buffer_size=20_000, batch_size=32, min_buffer_size=1_000,
            alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=700_000,
        ),
        "REINFORCE": lambda: REINFORCE(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=3e-4, gamma=0.99,
        ),
        "REINFORCEWithMeanBaseline": lambda: REINFORCEWithMeanBaseline(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=3e-4, gamma=0.99,
        ),
        "REINFORCEWithCriticBaseline": lambda: REINFORCEWithCriticBaseline(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=3e-4, lr_critic=3e-4, gamma=0.99,
        ),
        "PPO": lambda: PPO(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=3e-4, lr_critic=3e-4, gamma=0.99,
            clip_eps=0.2, n_epochs=4, entropy_coef=0.01, value_coef=0.5,
        ),
        "RandomRollout": lambda: RandomRollout(
            state_dim=27, n_actions=9, n_simulations=20, max_depth=20, gamma=1.0,
        ),
        "MCTS": lambda: MCTS(
            state_dim=27, n_actions=9, n_simulations=100, c_puct=1.41, max_depth=20, gamma=1.0,
        ),
        "AlphaZero": lambda: AlphaZero(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=3e-4, gamma=0.99, n_simulations=50, c_puct=1.0,
            temperature=1.0, l2_reg=1e-4,
            buffer_size=20_000, batch_size=64, min_buffer_size=1_000,
        ),
        "MuZero": lambda: MuZero(
            state_dim=27, n_actions=9, latent_dim=32, hidden_dims=[128, 64],
            lr=3e-4, gamma=0.99, n_simulations=50, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=20_000, batch_size=64, min_buffer_size=1_000,
        ),
        "MuZeroStochastic": lambda: MuZeroStochastic(
            state_dim=27, n_actions=9, latent_dim=32, n_chance=8, hidden_dims=[128, 64],
            lr=3e-4, gamma=0.99, n_simulations=50, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=20_000, batch_size=64, min_buffer_size=1_000,
        ),
        "ExpertApprentice": lambda: ExpertApprentice(
            state_dim=27, n_actions=9, hidden_dims=[128, 64],
            lr=3e-4, n_simulations=50, c_puct=1.41, gamma=1.0,
            max_depth=20, l2_reg=1e-4, use_student_ratio=0.0,
            buffer_size=20_000, batch_size=64, min_buffer_size=1_000,
        ),
    },
    # -----------------------------------------------------------------------
    # QUARTO — 114 dims, 32 actions, épisodes ~10-16 steps
    # epsilon_decay très lent (0.9999) → exploration sur ~46K épisodes
    # min_buffer_size = 10% du buffer pour diversité suffisante au démarrage
    # -----------------------------------------------------------------------
    "quarto": {
        "Random": lambda: RandomAgent(
            state_dim=114, n_actions=32,
        ),
        "DeepQLearning": lambda: DeepQLearning(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=3e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999,
        ),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=3e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999,
            target_update_freq=500,
        ),
        "DDQN_ER": lambda: DDQNWithExperienceReplay(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=3e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999,
            target_update_freq=500,
            buffer_size=50_000, batch_size=64, min_buffer_size=5_000,
        ),
        "DDQN_PER": lambda: DDQNWithPrioritizedExperienceReplay(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=3e-4, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999,
            target_update_freq=500,
            buffer_size=50_000, batch_size=64, min_buffer_size=5_000,
            alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=3_000_000,
        ),
        "REINFORCE": lambda: REINFORCE(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=1e-4, gamma=0.99,
        ),
        "REINFORCEWithMeanBaseline": lambda: REINFORCEWithMeanBaseline(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=1e-4, gamma=0.99,
        ),
        "REINFORCEWithCriticBaseline": lambda: REINFORCEWithCriticBaseline(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=1e-4, lr_critic=1e-4, gamma=0.99,
        ),
        "PPO": lambda: PPO(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=1e-4, lr_critic=1e-4, gamma=0.99,
            clip_eps=0.2, n_epochs=4, entropy_coef=0.01, value_coef=0.5,
        ),
        "RandomRollout": lambda: RandomRollout(
            state_dim=114, n_actions=32, n_simulations=10, max_depth=32, gamma=1.0,
        ),
        "MCTS": lambda: MCTS(
            state_dim=114, n_actions=32, n_simulations=50, c_puct=1.41, max_depth=32, gamma=1.0,
        ),
        "AlphaZero": lambda: AlphaZero(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=1e-4, gamma=0.99, n_simulations=25, c_puct=1.0,
            temperature=1.0, l2_reg=1e-4,
            buffer_size=50_000, batch_size=64, min_buffer_size=2_000,
        ),
        "MuZero": lambda: MuZero(
            state_dim=114, n_actions=32, latent_dim=64, hidden_dims=[256, 128],
            lr=1e-4, gamma=0.99, n_simulations=25, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=50_000, batch_size=64, min_buffer_size=2_000,
        ),
        "MuZeroStochastic": lambda: MuZeroStochastic(
            state_dim=114, n_actions=32, latent_dim=64, n_chance=8, hidden_dims=[256, 128],
            lr=1e-4, gamma=0.99, n_simulations=25, c_puct=1.0,
            n_unroll=5, temperature=1.0, l2_reg=1e-4,
            buffer_size=50_000, batch_size=64, min_buffer_size=2_000,
        ),
        "ExpertApprentice": lambda: ExpertApprentice(
            state_dim=114, n_actions=32, hidden_dims=[256, 128],
            lr=1e-4, n_simulations=25, c_puct=1.41, gamma=1.0,
            max_depth=32, l2_reg=1e-4, use_student_ratio=0.0,
            buffer_size=50_000, batch_size=64, min_buffer_size=2_000,
        ),
    },
}

# Agents sans entrainement (evalues directement)
NO_TRAINING_AGENTS = {"Random", "RandomRollout", "MCTS"}

# Agent par defaut pour chaque env (utilise par le GUI si --agent non specifie)
DEFAULT_AGENT = {
    "lineworld": "TabularQLearning",
    "gridworld": "TabularQLearning",
    "tictactoe": "DDQN_ER",
    "quarto": "DDQN_ER",
}

# Nombre d'episodes d'entrainement rapide (GUI, quand pas de modele)
QUICK_TRAIN_EPISODES = {
    "lineworld": 500,
    "gridworld": 1000,
    "tictactoe": 5000,
    "quarto": 5000,
}

# FPS par defaut pour le GUI (adapt au rythme du jeu)
DEFAULT_FPS = {
    "lineworld": 5,
    "gridworld": 5,
    "tictactoe": 2,
    "quarto": 2,
}


# ============================================================================
# FACTORIES D'ENVIRONNEMENTS
# ============================================================================

def make_env(env_name: str):
    """Cree l'environnement d'entrainement / evaluation (1 joueur ou VsRandom)."""
    if env_name == "lineworld":
        return LineWorld(size=5)
    elif env_name == "gridworld":
        return GridWorld.create_simple(size=5)
    elif env_name == "tictactoe":
        return TicTacToeVsRandom()
    elif env_name == "quarto":
        return QuartoVsRandom()
    raise ValueError(f"Environnement inconnu : {env_name}")


def make_env_2player(env_name: str):
    """Cree l'env 2 joueurs brut (self-play, GUI, imitation)."""
    if env_name == "tictactoe":
        return TicTacToe()
    elif env_name == "quarto":
        return Quarto()
    return make_env(env_name)


def find_latest_model(env_name: str, agent_name: str) -> str | None:
    """
    Cherche le dernier modele sauvegarde pour un agent.

    Ordre de recherche :
    1. results/latest/<env>/models/<agent>_ckpt*.pt  (dernier run)
    2. results/*/<env>/models/<agent>_ckpt*.pt        (n'importe quel run)

    Returns:
        Chemin du fichier .pt ou None si aucun modele trouve.
    """
    import glob
    import os

    safe_name = agent_name.replace(" ", "_").replace("/", "_")

    # 1. Chercher dans results/latest/
    pattern = os.path.join("results", "latest", env_name, "models", f"{safe_name}_ckpt*.pt")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]  # Dernier checkpoint (plus grand numero)

    # 2. Chercher dans tous les runs
    pattern = os.path.join("results", "*", env_name, "models", f"{safe_name}_ckpt*.pt")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]

    return None
