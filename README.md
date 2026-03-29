# DeepRL — Bibliothèque de Deep Reinforcement Learning

Bibliothèque pédagogique de Deep Reinforcement Learning en PyTorch, développée dans le cadre d'un projet universitaire.

## Objectif

Implémenter et comparer différentes techniques d'apprentissage par renforcement sur le jeu **Quarto** (jeu de stratégie à 2 joueurs), avec des environnements de test progressifs (LineWorld, GridWorld, TicTacToe).

## Installation

```bash
# Cloner et créer l'environnement
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

## Utilisation rapide

```bash
# Démos console (avec mesure parties/seconde)
python main.py --env quarto          # Random vs Random sur Quarto
python main.py --env tictactoe       # Random vs Random sur TicTacToe
python main.py --env gridworld       # Random vs Q-Learning sur GridWorld

# Interface graphique — observer un agent
python main.py --gui --env quarto

# Jouer contre l'IA
python main.py --play --env quarto

# Humain vs Humain
python main.py --pvp --env quarto

# Entraînement complet + benchmarks
python run_experiments.py
```

Voir [COMMANDS.md](COMMANDS.md) pour la liste complète des commandes et raccourcis GUI.

## Structure du projet

```
deeprl/
├── deeprl/
│   ├── envs/               # Environnements (LineWorld, GridWorld, TicTacToe, Quarto)
│   ├── agents/             # Agents (RandomAgent, HumanAgent, TabularQLearning)
│   ├── training/           # Trainer, Evaluator, Benchmark
│   └── gui/                # Interface Pygame (GameViewer, HumanVsAgentViewer)
├── main.py                 # Démos + GUI (--gui, --play, --pvp)
├── run_experiments.py       # Entraînement + métriques
├── SPECS.md                # Spécification technique complète
└── COMMANDS.md             # Référence des commandes
```

## Environnements

| Environnement | Type | State dim | Actions | Joueurs |
|---------------|------|-----------|---------|---------|
| LineWorld | Navigation 1D | 7 | 2 (←/→) | 1 |
| GridWorld | Navigation 2D | 25 | 4 (↑/↓/←/→) | 1 |
| TicTacToe | Morpion | 27 | 9 (positions) | 2 |
| **Quarto** | Jeu de stratégie | 114 | 32 (place + give) | 2 |

## Agents (Rendu 1)

| Agent | Catégorie | Description |
|-------|-----------|-------------|
| RandomAgent | Baseline | Action aléatoire uniforme |
| HumanAgent | Interactif | Joueur humain (console ou GUI) |
| TabularQLearning | Tabulaire | Q-Learning avec epsilon-greedy |

## Documentation

- **[SPECS.md](SPECS.md)** — Spécification technique : encoding des états/actions, interfaces, architecture
- **[COMMANDS.md](COMMANDS.md)** — Commandes CLI et raccourcis GUI

## Dépendances

- Python >= 3.9
- PyTorch >= 2.0
- NumPy >= 1.24
- Pygame >= 2.5
- Matplotlib >= 3.7

## Licence

MIT

## Metriques

- Score moyen apres N episodes d'entrainement
- Longueur moyenne des episodes
- Temps moyen par action

## Benchmarking

```bash
# Lancer un benchmark complet avec generation de graphiques
python main.py --benchmark

# Les resultats sont sauvegardes dans le dossier results/
```

## Interface Graphique

```bash
python -m deeprl.gui.game_viewer
```

## Licence

MIT
