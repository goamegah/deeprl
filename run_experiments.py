#!/usr/bin/env python3
"""
run_experiments.py — Exécution locale des expériences DeepRL

Entraîne tous les agents sur tous les environnements aux checkpoints demandés,
sauvegarde métriques (JSON), modèles (.pt) et génère les graphiques (PNG).
Aucun service externe (pas de wandb, pas de tensorboard).

Métriques mesurées (conformément aux consignes du projet) :
  - Score moyen (sur 100 épisodes d'évaluation)
  - Longueur moyenne des épisodes
  - Temps moyen par action (ms)
  - Taux de victoire (pour les jeux à 2 joueurs)

Usage :
    python run_experiments.py                              # Tout lancer
    python run_experiments.py --env gridworld               # Un seul env
    python run_experiments.py --agent TabularQLearning      # Un seul agent
    python run_experiments.py --checkpoints 1000,10000      # Checkpoints custom
    python run_experiments.py --resume results/<run_dir>    # Reprendre un run
    python run_experiments.py --plot results/<run_dir>      # Re-générer les plots

Sorties :
    results/<timestamp>/
        config.json               # Configuration du run
        <env>/
            metrics.json          # Métriques par agent × checkpoint
            training_curves.json  # Récompenses par épisode (courbes)
            models/<agent>.pt     # Poids sauvegardés
            plots/                # Graphiques PNG
        summary.csv               # Tableau récapitulatif global
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Matplotlib en mode non-interactif (serveur / CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from deeprl.envs import (
    LineWorld, GridWorld,
    TicTacToe, TicTacToeVsRandom,
    Quarto, QuartoVsRandom,
)
from deeprl.agents import (
    RandomAgent, TabularQLearning
)
from deeprl.training import Trainer, Evaluator


# ============================================================================
# CONSTANTES
# ============================================================================

DEFAULT_CHECKPOINTS = [1_000, 10_000, 100_000]
EVAL_EPISODES = 100
MAX_STEPS = 200

# Catégories d'agents (déterminent le mode d'entraînement)
NO_TRAINING_AGENTS = {"Random", "RandomRollout", "MCTS"}
ALPHAZERO_AGENTS = {"AlphaZero"}
IMITATION_AGENTS = {"ExpertApprentice_BC", "ExpertApprentice_DAgger"}


# ============================================================================
# REGISTRE DES AGENTS — hyperparamètres calibrés par environnement
# ============================================================================

AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "lineworld": {
        "Random":                     lambda: RandomAgent(state_dim=7, n_actions=2),
        "TabularQLearning":           lambda: TabularQLearning(n_states=7, n_actions=2, lr=0.1, gamma=0.99)
    },
    "gridworld": {
        "Random":                     lambda: RandomAgent(state_dim=25, n_actions=4),
        "TabularQLearning":           lambda: TabularQLearning(n_states=25, n_actions=4, lr=0.1, gamma=0.99)
    },
    "tictactoe": {
        "Random":                     lambda: RandomAgent(state_dim=27, n_actions=9)
    },
    "quarto": {
        "Random":                     lambda: RandomAgent(state_dim=114, n_actions=16)
    },
}


# ============================================================================
# CRÉATION D'ENVIRONNEMENTS
# ============================================================================

def make_env(env_name: str):
    """Crée l'environnement d'entraînement / évaluation."""
    if env_name == "lineworld":
        return LineWorld(size=7)
    elif env_name == "gridworld":
        return GridWorld.create_simple(size=5)
    elif env_name == "tictactoe":
        return TicTacToeVsRandom(use_onehot=True)
    elif env_name == "quarto":
        return QuartoVsRandom()
    raise ValueError(f"Environnement inconnu : {env_name}")


def make_env_2player(env_name: str):
    """Crée l'env 2 joueurs (self-play AlphaZero, imitation)."""
    if env_name == "tictactoe":
        return TicTacToe(use_onehot=True)
    elif env_name == "quarto":
        return Quarto()
    return make_env(env_name)


# ============================================================================
# BOUCLES D'ENTRAÎNEMENT SPÉCIALISÉES
# ============================================================================

def train_standard(env, agent, n_episodes: int) -> List[float]:
    """
    Entraînement standard (Q-Learning, DQN, REINFORCE, PPO, MuZero…).
    Utilise le Trainer qui passe env= à act() automatiquement.

    Returns:
        Liste des récompenses par épisode.
    """
    trainer = Trainer(env, agent, verbose=True, log_interval=max(100, n_episodes // 10))
    trainer.train(n_episodes=n_episodes, max_steps_per_episode=MAX_STEPS)
    return list(trainer.metrics.episode_rewards)


def train_alphazero(env_2player, agent, n_games: int) -> List[float]:
    """
    Entraînement AlphaZero par self-play.

    Joue n_games parties de self-play puis entraîne le réseau.

    Returns:
        Liste vide (pas de courbe épisodique standard).
    """
    print(f"    Self-play ({n_games} parties)…")
    examples = agent.self_play(env_2player, n_games=n_games, verbose=False)
    if examples:
        print(f"    Entraînement sur {len(examples)} exemples…")
        agent.train_on_examples(examples, n_epochs=10, batch_size=64)
    return []


def train_imitation(env_2player, agent, n_iterations: int) -> List[float]:
    """
    Entraînement par imitation (BC ou DAgger).

    Returns:
        Liste vide.
    """
    print(f"    Imitation ({agent.mode}, {n_iterations} itérations)…")
    agent.train(
        env_2player,
        n_iterations=n_iterations,
        episodes_per_iteration=20,
        epochs_per_iteration=50,
        verbose=False,
    )
    return []


# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_agent(env, agent, n_episodes: int = EVAL_EPISODES) -> Dict[str, float]:
    """
    Évalue un agent et retourne les métriques du projet.

    Returns:
        {mean_score, std_score, mean_length, std_length,
         mean_action_time, win_rate, loss_rate, draw_rate}
    """
    evaluator = Evaluator(env, agent, verbose=True)
    results = evaluator.evaluate(n_episodes=n_episodes, max_steps_per_episode=MAX_STEPS)
    summary = results.get_summary()

    return {
        "mean_score":       round(float(summary["mean_score"]), 4),
        "std_score":        round(float(summary["std_score"]), 4),
        "mean_length":      round(float(summary["mean_length"]), 2),
        "std_length":       round(float(summary["std_length"]), 2),
        "mean_action_time": round(float(summary["mean_action_time"]) * 1000, 4),  # ms
        "win_rate":         round(float(summary["win_rate"]), 4),
        "loss_rate":        round(float(summary["loss_rate"]), 4),
        "draw_rate":        round(float(summary["draw_rate"]), 4),
    }


# ============================================================================
# EXÉCUTION D'UNE EXPÉRIENCE (un env, tous ses agents)
# ============================================================================

def run_experiment(
    env_name: str,
    checkpoints: List[int],
    agent_filter: Optional[str],
    run_dir: str,
    resume_data: Optional[Dict] = None,
    eval_episodes: int = EVAL_EPISODES,
):
    """
    Lance les expériences pour un environnement donné.

    Pour chaque agent :
      1. Entraîne de manière incrémentale aux checkpoints
      2. Évalue à chaque checkpoint
      3. Sauvegarde le modèle

    Args:
        env_name:      Nom de l'environnement (lineworld, gridworld, …)
        checkpoints:   Liste croissante de checkpoints (nb épisodes)
        agent_filter:  Si fourni, ne lance que cet agent
        run_dir:       Dossier de résultats pour ce run
        resume_data:   Données d'un run précédent (pour --resume)
    """
    registry = AGENT_REGISTRY.get(env_name, {})
    if not registry:
        print(f"[ERREUR] Pas d'agents pour '{env_name}'")
        return {}

    # Filtrer les agents
    agent_names = list(registry.keys())
    if agent_filter:
        wanted = [a.strip() for a in agent_filter.split(",")]
        agent_names = [n for n in agent_names if n in wanted]
        if not agent_names:
            print(f"[ERREUR] Aucun agent parmi {wanted} n'est disponible pour '{env_name}'.")
            print(f"  Disponibles : {list(registry.keys())}")
            return {}

    # Dossiers
    env_dir = os.path.join(run_dir, env_name)
    models_dir = os.path.join(env_dir, "models")
    plots_dir = os.path.join(env_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Résultats
    all_metrics: Dict[str, Dict[int, Dict]] = {}
    all_curves: Dict[str, List[float]] = {}

    # Charger les résultats du resume (si applicable)
    if resume_data:
        all_metrics = resume_data.get("metrics", {})
        all_curves = resume_data.get("curves", {})

    sorted_ckpts = sorted(checkpoints)

    print(f"\n{'=' * 70}")
    print(f"  ENVIRONNEMENT : {env_name.upper()}")
    print(f"  Checkpoints : {sorted_ckpts}")
    print(f"  Agents : {len(agent_names)}")
    print(f"{'=' * 70}")

    for agent_name in agent_names:
        print(f"\n  ── {agent_name} {'─' * (50 - len(agent_name))}")

        # Vérifier si déjà terminé (resume)
        if agent_name in all_metrics:
            done_ckpts = set(int(k) for k in all_metrics[agent_name].keys())
            if done_ckpts >= set(sorted_ckpts):
                print(f"    [SKIP] Déjà terminé (resume)")
                continue

        # Créer l'agent
        agent = registry[agent_name]()
        env = make_env(env_name)

        # Déterminer le mode d'entraînement
        if agent_name in NO_TRAINING_AGENTS:
            training_mode = "none"
        elif agent_name in ALPHAZERO_AGENTS:
            training_mode = "alphazero"
        elif agent_name in IMITATION_AGENTS:
            training_mode = "imitation"
        else:
            training_mode = "standard"

        # Initialiser les structures
        if agent_name not in all_metrics:
            all_metrics[agent_name] = {}
        if agent_name not in all_curves:
            all_curves[agent_name] = []

        # Resume : trouver le dernier checkpoint complété
        done_ckpts = set(int(k) for k in all_metrics[agent_name].keys())
        remaining_ckpts = [c for c in sorted_ckpts if c not in done_ckpts]

        if not remaining_ckpts:
            print(f"    [SKIP] Tous les checkpoints sont faits")
            continue

        # Charger le modèle du dernier checkpoint (resume)
        if done_ckpts and agent_name not in NO_TRAINING_AGENTS:
            last_done = max(done_ckpts)
            # Essayer le nouveau format, puis l'ancien
            model_path = os.path.join(models_dir, f"{agent_name}_ckpt{last_done}.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(models_dir, f"{agent_name}.pt")
            if os.path.exists(model_path):
                print(f"    Chargement modèle (checkpoint {last_done})…")
                try:
                    agent.load(model_path)
                except Exception as e:
                    print(f"    [WARN] Impossible de charger : {e}")

        # ── Agents sans entraînement : évaluer une seule fois ──
        if training_mode == "none":
            print(f"    Évaluation ({eval_episodes} épisodes)…")
            metrics = evaluate_agent(env, agent, n_episodes=eval_episodes)
            for ckpt in sorted_ckpts:
                all_metrics[agent_name][str(ckpt)] = metrics
            _print_metrics(metrics)

        # ── Entraînement incrémental aux checkpoints ──
        else:
            prev_ckpt = max(done_ckpts) if done_ckpts else 0

            # Trainer réutilisé entre checkpoints (accumule les métriques)
            trainer = None
            if training_mode == "standard":
                trainer = Trainer(env, agent, verbose=True,
                                  log_interval=max(100, sorted_ckpts[0] // 10))

            for ckpt in remaining_ckpts:
                episodes_needed = ckpt - prev_ckpt
                t0 = time.time()

                print(f"    Entraînement → {ckpt:,} épisodes (+{episodes_needed:,})…")

                if training_mode == "standard":
                    trainer.train(n_episodes=episodes_needed,
                                  max_steps_per_episode=MAX_STEPS)
                    all_curves[agent_name] = list(trainer.metrics.episode_rewards)

                elif training_mode == "alphazero":
                    env_2p = make_env_2player(env_name)
                    train_alphazero(env_2p, agent, n_games=episodes_needed)

                elif training_mode == "imitation":
                    env_2p = make_env_2player(env_name)
                    n_iter = max(1, episodes_needed // 20)
                    train_imitation(env_2p, agent, n_iterations=n_iter)

                train_time = time.time() - t0

                # Évaluation
                print(f"    Évaluation au checkpoint {ckpt:,}…")
                metrics = evaluate_agent(env, agent, n_episodes=eval_episodes)
                metrics["training_time_s"] = round(train_time, 2)
                all_metrics[agent_name][str(ckpt)] = metrics

                _print_metrics(metrics)

                # Sauvegarder le modèle à chaque checkpoint (reproductibilité)
                if agent_name not in NO_TRAINING_AGENTS:
                    model_path = os.path.join(
                        models_dir, f"{agent_name}_ckpt{ckpt}.pt"
                    )
                    try:
                        agent.save(model_path)
                    except Exception as e:
                        print(f"    [WARN] Save échoué : {e}")

                prev_ckpt = ckpt

                # Sauvegarder les résultats intermédiaires (crash-safe)
                _save_json(os.path.join(env_dir, "metrics.json"), all_metrics)
                _save_json(os.path.join(env_dir, "training_curves.json"), all_curves)

    # Sauvegardes finales
    _save_json(os.path.join(env_dir, "metrics.json"), all_metrics)
    _save_json(os.path.join(env_dir, "training_curves.json"), all_curves)

    # Générer les graphiques
    print(f"\n  Génération des graphiques → {plots_dir}/")
    generate_plots(all_metrics, all_curves, env_name, plots_dir)

    return all_metrics


def _print_metrics(m: Dict):
    """Affiche un résumé d'évaluation en une ligne."""
    print(f"      score={m['mean_score']:.3f}±{m['std_score']:.3f}  "
          f"length={m['mean_length']:.1f}  "
          f"action_time={m['mean_action_time']:.3f}ms  "
          f"win={m['win_rate']:.0%}")


# ============================================================================
# STOCKAGE LOCAL (JSON)
# ============================================================================

def _save_json(path: str, data: Any):
    """Sauvegarde un objet en JSON (atomique via rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _load_json(path: str) -> Any:
    """Charge un fichier JSON."""
    with open(path) as f:
        return json.load(f)


# ============================================================================
# TABLEAU RÉCAPITULATIF (terminal + CSV)
# ============================================================================

def print_summary_table(all_results: Dict[str, Dict], checkpoints: List[int]):
    """
    Affiche un tableau récapitulatif formaté dans le terminal.

    Args:
        all_results: {env_name: {agent_name: {checkpoint: metrics}}}
        checkpoints: Liste de checkpoints
    """
    for env_name, agents in all_results.items():
        print(f"\n{'═' * 100}")
        print(f"  {env_name.upper()}")
        print(f"{'═' * 100}")

        header = f"  {'Agent':<28} {'Checkpoint':>10}  {'Score moyen':>18}  {'Long. moy.':>10}  {'Tps/action':>10}  {'Win %':>6}"
        print(header)
        print(f"  {'─' * 96}")

        for agent_name, ckpt_data in agents.items():
            first = True
            for ckpt in sorted(checkpoints):
                key = str(ckpt)
                if key not in ckpt_data:
                    continue
                m = ckpt_data[key]
                name_col = agent_name if first else ""
                score_str = f"{m['mean_score']:+.3f} ± {m['std_score']:.3f}"
                length_str = f"{m['mean_length']:.1f}"
                time_str = f"{m['mean_action_time']:.3f}ms"
                win_str = f"{m['win_rate']:.0%}"
                print(f"  {name_col:<28} {ckpt:>10,}  {score_str:>18}  {length_str:>10}  {time_str:>10}  {win_str:>6}")
                first = False
            print(f"  {'─' * 96}")


def generate_csv(all_results: Dict[str, Dict], path: str):
    """Exporte les résultats en CSV."""
    lines = ["env,agent,checkpoint,mean_score,std_score,mean_length,std_length,mean_action_time_ms,win_rate"]
    for env_name, agents in all_results.items():
        for agent_name, ckpt_data in agents.items():
            for ckpt_str, m in sorted(ckpt_data.items(), key=lambda x: int(x[0])):
                lines.append(
                    f"{env_name},{agent_name},{ckpt_str},"
                    f"{m['mean_score']},{m['std_score']},"
                    f"{m['mean_length']},{m['std_length']},"
                    f"{m['mean_action_time']},{m['win_rate']}"
                )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  CSV exporté → {path}")


# ============================================================================
# GRAPHIQUES (matplotlib)
# ============================================================================

# Style global
PLOT_STYLE = {
    "figure.figsize": (12, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}

# Palette de couleurs distinctes
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]


def generate_plots(
    metrics: Dict[str, Dict[int, Dict]],
    curves: Dict[str, List[float]],
    env_name: str,
    plots_dir: str,
):
    """Génère tous les graphiques pour un environnement."""
    plt.rcParams.update(PLOT_STYLE)

    _plot_comparison_bars(metrics, env_name, "mean_score", "Score moyen",
                          os.path.join(plots_dir, "comparison_score.png"))
    _plot_comparison_bars(metrics, env_name, "mean_length", "Longueur moyenne",
                          os.path.join(plots_dir, "comparison_length.png"))
    _plot_action_times(metrics, env_name,
                       os.path.join(plots_dir, "action_times.png"))
    if curves:
        _plot_learning_curves(curves, env_name,
                              os.path.join(plots_dir, "learning_curves.png"))

    _plot_summary_table_image(metrics, env_name,
                              os.path.join(plots_dir, "summary_table.png"))


def _plot_comparison_bars(
    metrics: Dict, env_name: str, metric_key: str, metric_label: str, path: str,
):
    """
    Barres groupées : un groupe par checkpoint, une barre par agent.
    """
    agents = list(metrics.keys())
    ckpts = sorted({int(c) for a in metrics.values() for c in a.keys()})

    if not agents or not ckpts:
        return

    n_agents = len(agents)
    n_ckpts = len(ckpts)
    bar_width = 0.8 / n_agents
    x = np.arange(n_ckpts)

    fig, ax = plt.subplots(figsize=(max(10, n_ckpts * 2.5), 6))

    for i, agent_name in enumerate(agents):
        values = []
        errors = []
        for ckpt in ckpts:
            key = str(ckpt)
            if key in metrics[agent_name]:
                values.append(metrics[agent_name][key][metric_key])
                if metric_key == "mean_score":
                    errors.append(metrics[agent_name][key].get("std_score", 0))
                elif metric_key == "mean_length":
                    errors.append(metrics[agent_name][key].get("std_length", 0))
                else:
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)

        color = COLORS[i % len(COLORS)]
        positions = x + i * bar_width
        bars = ax.bar(positions, values, bar_width, label=agent_name,
                   color=color, alpha=0.85)

        # Annotate each bar with its value
        for bar_rect, val in zip(bars, values):
            if val >= 0:
                y_pos = val + 0.03
                va = "bottom"
            else:
                y_pos = val - 0.03
                va = "top"
            ax.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                y_pos,
                f"{val:.2f}",
                ha="center", va=va,
                fontsize=6.5, fontweight="bold",
                rotation=0,
            )

    ax.set_xlabel("Checkpoint (épisodes d'entraînement)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{env_name.upper()} — {metric_label} par agent et checkpoint")
    ax.set_xticks(x + bar_width * (n_agents - 1) / 2)
    ax.set_xticklabels([f"{c:,}" for c in ckpts])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_action_times(metrics: Dict, env_name: str, path: str):
    """Barres horizontales : temps moyen par action (dernier checkpoint)."""
    agents = list(metrics.keys())
    if not agents:
        return

    # Prendre le dernier checkpoint disponible pour chaque agent
    times = []
    names = []
    for agent_name in agents:
        ckpts = sorted(metrics[agent_name].keys(), key=int)
        if ckpts:
            t = metrics[agent_name][ckpts[-1]]["mean_action_time"]
            times.append(t)
            names.append(agent_name)

    if not times:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    y = np.arange(len(names))
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]

    ax.barh(y, times, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Temps moyen par action (ms)")
    ax.set_title(f"{env_name.upper()} — Temps de décision par agent")
    ax.invert_yaxis()

    # Annoter les valeurs
    for i, v in enumerate(times):
        ax.text(v + max(times) * 0.01, i, f"{v:.3f}ms", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curves(curves: Dict[str, List[float]], env_name: str, path: str):
    """
    Courbes d'apprentissage (récompense par épisode, lissées).
    """
    has_data = False
    fig, ax = plt.subplots(figsize=(12, 6))

    all_smoothed = []  # Track smoothed values for y-axis limits

    for i, (agent_name, rewards) in enumerate(curves.items()):
        if not rewards:
            continue
        has_data = True
        episodes = np.arange(1, len(rewards) + 1)
        # Lissage par moyenne glissante (fenêtre = 2% des épisodes, min 10)
        window = max(10, len(rewards) // 50)
        smoothed = _rolling_mean(rewards, window)
        std = _rolling_std(rewards, window)
        color = COLORS[i % len(COLORS)]

        ax.plot(episodes, smoothed, label=agent_name, color=color, alpha=0.9)
        # Zone claire autour (±0.5 std) — narrower band for readability
        ax.fill_between(episodes, smoothed - 0.5 * std, smoothed + 0.5 * std,
                         color=color, alpha=0.12)
        all_smoothed.extend(smoothed)

    if not has_data:
        plt.close(fig)
        return

    # Clamp y-axis to meaningful range (based on smoothed data, not std bands)
    if all_smoothed:
        y_min = min(all_smoothed)
        y_max = max(all_smoothed)
        margin = max(0.2, (y_max - y_min) * 0.15)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense (lissée)")
    ax.set_title(f"{env_name.upper()} — Courbes d'apprentissage")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_table_image(metrics: Dict, env_name: str, path: str):
    """Rend le tableau de résultats en image PNG (pour le rapport)."""
    agents = list(metrics.keys())
    ckpts = sorted({int(c) for a in metrics.values() for c in a.keys()})

    if not agents or not ckpts:
        return

    # Construire les données du tableau
    col_labels = ["Agent"] + [f"{c:,} ep." for c in ckpts]
    cell_text = []

    for agent_name in agents:
        row = [agent_name]
        for ckpt in ckpts:
            key = str(ckpt)
            if key in metrics[agent_name]:
                m = metrics[agent_name][key]
                row.append(f"{m['mean_score']:+.3f}\n±{m['std_score']:.3f}")
            else:
                row.append("—")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(ckpts) * 3), max(4, len(agents) * 0.6)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style de l'en-tête
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alterner les couleurs des lignes
    for i in range(len(agents)):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title(f"{env_name.upper()} — Score moyen par agent et checkpoint",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _rolling_mean(data: list, window: int) -> np.ndarray:
    """Moyenne glissante."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _rolling_std(data: list, window: int) -> np.ndarray:
    """Écart-type glissant (même fenêtre et padding que _rolling_mean)."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return np.zeros_like(arr)
    # Edge-pad identically to _rolling_mean for consistency
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    # Vectorised rolling std via cumulative sums (prepend 0 for correct length)
    cum = np.concatenate(([0.0], np.cumsum(padded)))
    cum2 = np.concatenate(([0.0], np.cumsum(padded ** 2)))
    s = cum[window:] - cum[:-window]
    s2 = cum2[window:] - cum2[:-window]
    variance = s2 / window - (s / window) ** 2
    # Clamp numerical noise to zero before sqrt
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepRL — Exécution des expériences et génération des résultats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python run_experiments.py                              # Tout lancer
  python run_experiments.py --env gridworld              # Un seul env
  python run_experiments.py --env tictactoe --agent MCTS # Un agent
  python run_experiments.py --checkpoints 1000,10000     # Checkpoints custom
  python run_experiments.py --resume results/2024-01-15  # Reprendre
  python run_experiments.py --plot results/2024-01-15    # Re-générer plots
  python run_experiments.py --eval results/2024-01-15    # Reproduire résultats
        """,
    )
    parser.add_argument("--env", type=str, default=None,
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Environnement cible (défaut : tous)")
    parser.add_argument("--agent", type=str, default=None,
                        help="Agent cible (défaut : tous)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Checkpoints séparés par des virgules (ex: 1000,10000)")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help=f"Épisodes d'évaluation par checkpoint (défaut: {EVAL_EPISODES})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin d'un run à reprendre")
    parser.add_argument("--plot", type=str, default=None,
                        help="Chemin d'un run existant pour re-générer les graphiques")
    parser.add_argument("--eval", type=str, default=None,
                        help="Chemin d'un run existant pour reproduire les résultats "
                             "(charge les modèles et évalue sans ré-entraîner)")
    args = parser.parse_args()

    # ── Mode re-plot uniquement ──
    if args.plot:
        replot(args.plot, agent_filter=args.agent, env_filter=args.env)
        return

    # ── Mode évaluation / reproductibilité ──
    if args.eval:
        ckpt_filter = None
        if args.checkpoints:
            ckpt_filter = [int(c.strip()) for c in args.checkpoints.split(",")]
        reproduce(args.eval, agent_filter=args.agent, env_filter=args.env,
                  eval_episodes=args.eval_episodes, checkpoints=ckpt_filter)
        return

    # ── Configuration ──
    checkpoints = DEFAULT_CHECKPOINTS
    if args.checkpoints:
        checkpoints = sorted(int(c.strip()) for c in args.checkpoints.split(","))

    envs = [args.env] if args.env else list(AGENT_REGISTRY.keys())

    # ── Dossier de sortie ──
    if args.resume:
        run_dir = args.resume
        print(f"\n  Reprise du run : {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("results", timestamp)
        os.makedirs(run_dir, exist_ok=True)

    # Sauvegarder la config
    config = {
        "timestamp": datetime.now().isoformat(),
        "envs": envs,
        "checkpoints": checkpoints,
        "eval_episodes": args.eval_episodes,
        "max_steps": MAX_STEPS,
        "agent_filter": args.agent,
    }
    _save_json(os.path.join(run_dir, "config.json"), config)

    print(f"\n{'═' * 70}")
    print(f"  DeepRL — Experiment Runner")
    print(f"  Résultats → {run_dir}")
    print(f"  Envs : {envs}")
    print(f"  Checkpoints : {checkpoints}")
    print(f"  Eval episodes : {args.eval_episodes}")
    print(f"{'═' * 70}")

    # ── Lancer les expériences ──
    all_results = {}

    for env_name in envs:
        # Charger les données de resume si disponibles
        resume_data = None
        if args.resume:
            metrics_path = os.path.join(run_dir, env_name, "metrics.json")
            curves_path = os.path.join(run_dir, env_name, "training_curves.json")
            if os.path.exists(metrics_path):
                resume_data = {
                    "metrics": _load_json(metrics_path),
                    "curves": _load_json(curves_path) if os.path.exists(curves_path) else {},
                }
                print(f"\n  [RESUME] Données chargées pour {env_name}")

        env_metrics = run_experiment(
            env_name=env_name,
            checkpoints=checkpoints,
            agent_filter=args.agent,
            run_dir=run_dir,
            resume_data=resume_data,
            eval_episodes=args.eval_episodes,
        )
        if env_metrics:
            all_results[env_name] = env_metrics

    # ── Tableau récapitulatif ──
    if all_results:
        print_summary_table(all_results, checkpoints)
        csv_path = os.path.join(run_dir, "summary.csv")
        generate_csv(all_results, csv_path)

    # ── Lien symbolique latest ──
    latest = os.path.join("results", "latest")
    try:
        if os.path.islink(latest):
            os.unlink(latest)
        os.symlink(os.path.abspath(run_dir), latest)
    except OSError:
        pass

    print(f"\n{'═' * 70}")
    print(f"  Terminé ! Résultats dans : {run_dir}")
    print(f"{'═' * 70}\n")


def reproduce(run_dir: str, agent_filter: Optional[str] = None,
              env_filter: Optional[str] = None, eval_episodes: int = EVAL_EPISODES,
              checkpoints: Optional[List[int]] = None):
    """Charge les modèles sauvegardés et les évalue (reproductibilité).

    Permet au prof de vérifier les résultats sans ré-entraîner.
    Charge chaque <agent>_ckpt<N>.pt, l'évalue, et affiche le score.
    """
    print(f"\n{'═' * 70}")
    print(f"  DeepRL — Reproduction des résultats")
    print(f"  Run source : {run_dir}")
    print(f"  Épisodes d'évaluation : {eval_episodes}")
    if checkpoints:
        print(f"  Checkpoints : {checkpoints}")
    print(f"{'═' * 70}")

    wanted_agents = None
    if agent_filter:
        wanted_agents = set(a.strip() for a in agent_filter.split(","))
        print(f"  Agents : {sorted(wanted_agents)}")

    env_names = [env_filter] if env_filter else list(AGENT_REGISTRY.keys())
    all_results = {}

    for env_name in env_names:
        models_dir = os.path.join(run_dir, env_name, "models")
        if not os.path.exists(models_dir):
            continue

        # Découvrir les fichiers .pt
        pt_files = sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))
        if not pt_files:
            continue

        registry = AGENT_REGISTRY.get(env_name, {})
        env = make_env(env_name)

        print(f"\n{'=' * 70}")
        print(f"  ENVIRONNEMENT : {env_name.upper()}")
        print(f"{'=' * 70}")

        env_metrics = {}

        for pt_file in pt_files:
            # Parse : <AgentName>_ckpt<N>.pt
            base = pt_file.replace(".pt", "")
            parts = base.rsplit("_ckpt", 1)
            if len(parts) != 2:
                # Ancien format <AgentName>.pt (pas de checkpoint)
                agent_name = base
                ckpt_str = "final"
            else:
                agent_name, ckpt_str = parts[0], parts[1]

            # Filtre par checkpoint
            if checkpoints and ckpt_str != "final":
                try:
                    if int(ckpt_str) not in checkpoints:
                        continue
                except ValueError:
                    pass

            if wanted_agents and agent_name not in wanted_agents:
                continue
            if agent_name not in registry:
                print(f"  [SKIP] {agent_name} — pas dans le registre de {env_name}")
                continue

            # Créer un agent frais et charger les poids
            try:
                agent = registry[agent_name]()
                agent.load(os.path.join(models_dir, pt_file))
            except Exception as e:
                print(f"  [ERREUR] {pt_file} : {e}")
                continue

            # Évaluer
            print(f"\n  ── {agent_name} (ckpt {ckpt_str}) ──")
            metrics = evaluate_agent(env, agent, n_episodes=eval_episodes)
            _print_metrics(metrics)

            if agent_name not in env_metrics:
                env_metrics[agent_name] = {}
            env_metrics[agent_name][ckpt_str] = metrics

        if env_metrics:
            all_results[env_name] = env_metrics

    # Tableau récapitulatif
    if all_results:
        # Récupérer les checkpoints depuis la config
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            config = _load_json(config_path)
            ckpts = config.get("checkpoints", DEFAULT_CHECKPOINTS)
        else:
            ckpts = DEFAULT_CHECKPOINTS
        print_summary_table(all_results, ckpts)

    print(f"\n{'═' * 70}")
    print(f"  Reproduction terminée !")
    print(f"{'═' * 70}\n")


def replot(run_dir: str, agent_filter: Optional[str] = None,
           env_filter: Optional[str] = None):
    """Re-génère les graphiques à partir des données sauvegardées.

    Args:
        run_dir: Chemin du run existant.
        agent_filter: Agents à afficher, séparés par des virgules.
                      Si None, tous les agents sont affichés.
        env_filter: Environnement cible. Si None, tous les envs.
    """
    wanted = None
    if agent_filter:
        wanted = set(a.strip() for a in agent_filter.split(","))

    print(f"\n  Re-génération des graphiques depuis {run_dir}")
    if wanted:
        print(f"  Agents filtrés : {sorted(wanted)}")
    if env_filter:
        print(f"  Env filtré : {env_filter}")

    all_results = {}

    env_names = [env_filter] if env_filter else list(AGENT_REGISTRY.keys())
    for env_name in env_names:
        env_dir = os.path.join(run_dir, env_name)
        metrics_path = os.path.join(env_dir, "metrics.json")
        curves_path = os.path.join(env_dir, "training_curves.json")
        plots_dir = os.path.join(env_dir, "plots")

        if not os.path.exists(metrics_path):
            continue

        metrics = _load_json(metrics_path)
        curves = _load_json(curves_path) if os.path.exists(curves_path) else {}

        # Filtrer les agents si demandé
        if wanted:
            metrics = {k: v for k, v in metrics.items() if k in wanted}
            curves = {k: v for k, v in curves.items() if k in wanted}
            if not metrics:
                print(f"    {env_name} : aucun agent correspondant")
                continue

        os.makedirs(plots_dir, exist_ok=True)
        generate_plots(metrics, curves, env_name, plots_dir)
        all_results[env_name] = metrics
        print(f"    {env_name} : {len(metrics)} agents")

    if all_results:
        # Trouver les checkpoints depuis la config
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            config = _load_json(config_path)
            ckpts = config.get("checkpoints", DEFAULT_CHECKPOINTS)
        else:
            ckpts = DEFAULT_CHECKPOINTS
        print_summary_table(all_results, ckpts)

    print(f"\n  Plots re-générés !")


if __name__ == "__main__":
    main()
