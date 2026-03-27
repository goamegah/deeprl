# DeepRL - Bibliotheque de Deep Reinforcement Learning

Une bibliotheque pedagogique de Deep Reinforcement Learning avec PyTorch.

## Objectif

Evaluer et comparer differentes techniques d'apprentissage par renforcement profond sur plusieurs environnements.

## Structure du Projet

```
deeprl/
├── envs/                    # Environnements
│   ├── base.py             # Interface abstraite Environment
│   ├── line_world.py       # LineWorld (1D)
│   ├── grid_world.py       # GridWorld (2D)
│   ├── tictactoe.py        # TicTacToe
│   └── quarto.py           # Quarto
├── agents/                  # Agents
│   ├── base.py             # Interface abstraite Agent
│   ├── random_agent.py     # Agent aleatoire
│   ├── tabular/            # Methodes tabulaires
│   │   └── q_learning.py
│   ├── value_based/        # Methodes basees sur la valeur
│   │   ├── dqn.py
│   │   ├── ddqn.py
│   │   └── prioritized_replay.py
│   ├── policy_based/       # Methodes basees sur la politique
│   │   ├── reinforce.py
│   │   └── ppo.py
│   └── planning/           # Methodes de planification
│       ├── mcts.py
│       ├── alphazero.py
│       └── muzero.py
├── networks/               # Reseaux de neurones
│   ├── mlp.py             # Multi-Layer Perceptron
│   └── shared.py          # Reseaux partages Actor-Critic
├── memory/                 # Memoires de replay
│   ├── replay_buffer.py
│   └── prioritized_buffer.py
├── training/              # Boucles d'entrainement
│   ├── trainer.py
│   ├── evaluator.py
│   └── benchmark.py       # Systeme de benchmarking avec graphiques
├── utils/                 # Utilitaires
│   ├── metrics.py
│   └── visualization.py
├── gui/                   # Interface graphique
│   └── game_viewer.py
└── main.py               # Point d'entree
```

## Installation

```bash
# Creer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer les dependances
pip install -r requirements.txt
```

## Environnements

| Environnement | Type | Description |
|--------------|------|-------------|
| LineWorld | Test | Monde 1D simple |
| GridWorld | Test | Monde 2D avec obstacles |
| TicTacToe | Test | Jeu a 2 joueurs |
| Quarto | Avance | Jeu de plateau complexe |

## Agents

### Agents Simples
- **Random** : Actions aleatoires

### Methodes Tabulaires
- **TabularQLearning** : Q-Learning classique

### Value-Based (Deep)
- **DQN** : Deep Q-Network
- **DDQN** : Double DQN
- **DDQN + Experience Replay**
- **DDQN + Prioritized Experience Replay**

### Policy-Based
- **REINFORCE** : Policy Gradient simple
- **REINFORCE + Baseline**
- **Actor-Critic (A2C)**
- **PPO** : Proximal Policy Optimization

### Planning
- **Random Rollout**
- **MCTS (UCT)**
- **Expert Apprentice**
- **AlphaZero**
- **MuZero**

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
