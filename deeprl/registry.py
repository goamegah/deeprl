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
)


# ============================================================================
# REGISTRE DES AGENTS — hyperparametres calibres par environnement
# ============================================================================

AGENT_REGISTRY = {
    "lineworld": {
        "Random":              lambda: RandomAgent(state_dim=7, n_actions=2),
        "TabularQLearning":    lambda: TabularQLearning(n_states=7, n_actions=2, lr=0.1, gamma=0.99),
        "DeepQLearning":       lambda: DeepQLearning(state_dim=7, n_actions=2, hidden_dims=[32]),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(state_dim=7, n_actions=2, hidden_dims=[32], target_update_freq=50),
        "DDQN_ER":             lambda: DDQNWithExperienceReplay(state_dim=7, n_actions=2, hidden_dims=[32], target_update_freq=50, buffer_size=5000),
        "DDQN_PER":            lambda: DDQNWithPrioritizedExperienceReplay(state_dim=7, n_actions=2, hidden_dims=[32], target_update_freq=50, buffer_size=5000),
    },
    "gridworld": {
        "Random":              lambda: RandomAgent(state_dim=25, n_actions=4),
        "TabularQLearning":    lambda: TabularQLearning(n_states=25, n_actions=4, lr=0.1, gamma=0.99),
        "DeepQLearning":       lambda: DeepQLearning(state_dim=25, n_actions=4, hidden_dims=[64, 32]),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(state_dim=25, n_actions=4, hidden_dims=[64, 32], target_update_freq=100),
        "DDQN_ER":             lambda: DDQNWithExperienceReplay(state_dim=25, n_actions=4, hidden_dims=[64, 32], target_update_freq=100, buffer_size=10000),
        "DDQN_PER":            lambda: DDQNWithPrioritizedExperienceReplay(state_dim=25, n_actions=4, hidden_dims=[64, 32], target_update_freq=100, buffer_size=10000),
    },
    "tictactoe": {
        "Random":              lambda: RandomAgent(state_dim=27, n_actions=9),
        "DeepQLearning":       lambda: DeepQLearning(state_dim=27, n_actions=9, hidden_dims=[128, 64], lr=5e-4, epsilon_decay=0.9995),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(state_dim=27, n_actions=9, hidden_dims=[128, 64], lr=5e-4, epsilon_decay=0.9995, target_update_freq=200),
        "DDQN_ER":             lambda: DDQNWithExperienceReplay(state_dim=27, n_actions=9, hidden_dims=[128, 64], lr=5e-4, epsilon_decay=0.9995, target_update_freq=200, buffer_size=20000),
        "DDQN_PER":            lambda: DDQNWithPrioritizedExperienceReplay(state_dim=27, n_actions=9, hidden_dims=[128, 64], lr=5e-4, epsilon_decay=0.9995, target_update_freq=200, buffer_size=20000),
    },
    "quarto": {
        "Random":              lambda: RandomAgent(state_dim=114, n_actions=32),
        "DeepQLearning":       lambda: DeepQLearning(state_dim=114, n_actions=32, hidden_dims=[256, 128], lr=3e-4, epsilon_decay=0.9999),
        "DoubleDeepQLearning": lambda: DoubleDeepQLearning(state_dim=114, n_actions=32, hidden_dims=[256, 128], lr=3e-4, epsilon_decay=0.9999, target_update_freq=500),
        "DDQN_ER":             lambda: DDQNWithExperienceReplay(state_dim=114, n_actions=32, hidden_dims=[256, 128], lr=3e-4, epsilon_decay=0.9999, target_update_freq=500, buffer_size=50000),
        "DDQN_PER":            lambda: DDQNWithPrioritizedExperienceReplay(state_dim=114, n_actions=32, hidden_dims=[256, 128], lr=3e-4, epsilon_decay=0.9999, target_update_freq=500, buffer_size=50000),
    },
}

# Agents sans entrainement (evalues directement)
NO_TRAINING_AGENTS = {"Random"}

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
        return LineWorld(size=7)
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
