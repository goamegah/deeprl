# DeepRL — Bibliothèque de Deep Reinforcement Learning

Bibliothèque pédagogique de Deep Reinforcement Learning en PyTorch, développée dans le cadre d'un projet universitaire.

## Objectif

Implémenter et comparer différentes techniques d'apprentissage par renforcement sur le jeu **Quarto** (jeu de stratégie à 2 joueurs), avec des environnements de test progressifs (LineWorld, GridWorld, TicTacToe).

## Installation

```bash
git clone <repo_url> && cd deeprl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ou avec `uv` :

```bash
uv sync
source .venv/bin/activate
```

---

## Démonstration (`main.py`)

Trois flags suffisent : `--env`, `--agent`, `--versus`.

| Règle | Comportement |
|-------|-------------|
| Sans `--agent` | Démo texte dans le terminal (stats, parties/sec) |
| Avec `--agent` | Ouvre l'interface graphique Pygame |
| `Human` comme valeur | Vous jouez avec la souris |

### Démo texte (terminal)

```bash
python main.py --env lineworld
python main.py --env gridworld
python main.py --env tictactoe
python main.py --env quarto
```

### Interface graphique — observer un agent face à Random

```bash
python main.py --env tictactoe --agent DDQN_ER
python main.py --env quarto    --agent DDQN_PER
```

### Interface graphique — observer deux agents s'affronter

```bash
python main.py --env tictactoe --agent DDQN_ER --versus DDQN_PER
python main.py --env quarto    --agent DDQN_ER --versus Random
```

### Jouer contre un agent

```bash
python main.py --env tictactoe --agent Human --versus DDQN_ER
python main.py --env quarto    --agent Human --versus DDQN_PER
```

### Humain vs Humain

```bash
python main.py --env tictactoe --agent Human --versus Human
python main.py --env quarto    --agent Human --versus Human
```

### Contrôles GUI

| Touche | Action |
|--------|--------|
| `SPACE` | Pause / Reprendre |
| `N` | Avancer step par step |
| `+` / `-` | Augmenter / réduire la vitesse |
| `R` | Relancer une partie |
| `F11` | Plein écran |
| `ESC` | Quitter |

---

## Entraînement et benchmarks (`run_experiments.py`)

Entraîne tous les agents, évalue aux checkpoints 1K/10K/100K/1M épisodes, sauvegarde les métriques, modèles et graphiques.

```bash
# Tout lancer
python run_experiments.py

# Un seul environnement
python run_experiments.py --env tictactoe

# Un seul agent
python run_experiments.py --env tictactoe --agent DDQN_ER

# Checkpoints personnalisés
python run_experiments.py --checkpoints 1000,10000,100000

# Reprendre un run interrompu
python run_experiments.py --resume results/2026-04-24_11-25-20

# Re-générer les graphiques uniquement
python run_experiments.py --plot results/2026-04-24_11-25-20

# Reproduire les résultats depuis les modèles sauvegardés
python run_experiments.py --eval results/2026-04-24_11-25-20
```

### Sorties

```
results/<timestamp>/
    config.json
    summary.csv
    <env>/
        metrics.json          # Métriques par agent × checkpoint
        training_curves.json  # Récompenses par épisode
        models/               # Poids .pt par agent et checkpoint
        plots/                # Graphiques PNG
```

---

## Structure du projet

```
deeprl/
├── deeprl/
│   ├── registry.py      # Source unique : agents disponibles, hyperparamètres calibrés
│   │                    # par env, make_env/make_env_2player, find_latest_model
│   ├── envs/            # LineWorld, GridWorld, TicTacToe, Quarto
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── human_agent.py
│   │   ├── tabular/         # TabularQLearning
│   │   ├── value_based/     # DeepQLearning → DDQN_PER
│   │   ├── policy_gradient/ # REINFORCE, PPO
│   │   ├── tree_search/     # RandomRollout, MCTS, AlphaZero, MuZero
│   │   └── imitation/       # ExpertApprentice
│   ├── networks/        # MLP (Xavier init)
│   ├── memory/          # ReplayBuffer, PrioritizedReplayBuffer
│   ├── training/        # Trainer, Evaluator
│   └── gui/             # GameViewer, AgentVsAgentViewer, HumanVsAgentViewer
├── main.py              # Démo + GUI
├── run_experiments.py   # Entraînement + métriques
└── results/             # Résultats sauvegardés
```

---

## Environnements

| Environnement | State dim | Actions | Joueurs |
|---------------|-----------|---------|---------|
| LineWorld | 5 | 2 (←/→) | 1 |
| GridWorld | 25 | 4 (↑↓←→) | 1 |
| TicTacToe | 27 | 9 | 2 |
| **Quarto** | 114 | 32 | 2 |

## Agents

| Agent | Catégorie |
|-------|-----------|
| RandomAgent | Baseline |
| HumanAgent | Interactif |
| TabularQLearning | Tabulaire |
| DeepQLearning | Value-based |
| DoubleDeepQLearning | Value-based |
| DDQN_ER | Value-based + Experience Replay |
| DDQN_PER | Value-based + Prioritized ER |
| REINFORCE, +Baseline Mean, +Baseline Critic, PPO | Policy Gradient |
| RandomRollout, MCTS | Tree Search |
| AlphaZero, MuZero, MuZero Stochastique | Tree Search + Neural |
| ExpertApprentice | Imitation Learning |

## Métriques

Évaluées sur **1 000 épisodes** à chaque checkpoint (politique d'évaluation, pas d'entraînement) :

- Score moyen ± écart-type
- Longueur moyenne des épisodes
- Temps moyen par action (ms)
- Taux de victoire / défaite / nul

## Dépendances

- Python >= 3.9
- PyTorch >= 2.0
- NumPy >= 1.24
- Pygame >= 2.5
- Matplotlib >= 3.7



