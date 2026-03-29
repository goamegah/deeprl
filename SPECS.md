# DeepRL — Spécification technique (Rendu 1)

## Table des matières

1. [Gameplay choisi et environnements](#1-gameplay-choisi-et-environnements)
2. [Simulation random et parties/seconde](#2-simulation-random-et-partiesseconde)
3. [Jeu avec joueur humain + GUI](#3-jeu-avec-joueur-humain--gui)
4. [Description de l'état du jeu (encoding)](#4-description-de-létat-du-jeu-encoding)
5. [Description d'une action du jeu (encoding)](#5-description-dune-action-du-jeu-encoding)
6. [Architecture du projet](#6-architecture-du-projet)
7. [Interfaces abstraites](#7-interfaces-abstraites)
8. [Détails des agents implémentés](#8-détails-des-agents-implémentés)
9. [Infrastructure d'entraînement](#9-infrastructure-dentraînement)
10. [Dépendances](#10-dépendances)

---

## 1. Gameplay choisi et environnements

### Jeu principal : Quarto

**Quarto** est un jeu de stratégie à 2 joueurs sur un plateau 4x4 avec 16 pièces uniques. Chaque pièce possède 4 attributs binaires :

| Attribut | Valeurs |
|----------|---------|
| Taille | Grand (`tall`) / Petit (`short`) |
| Couleur | Foncé (`dark`) / Clair (`light`) |
| Remplissage | Plein (`solid`) / Creux (`hollow`) |
| Forme | Carré (`square`) / Rond (`round`) |

**Règle :** Un joueur **choisit** une pièce et la **donne** à son adversaire, qui doit ensuite la **placer** sur une case libre. Le premier à aligner 4 pièces partageant au moins 1 attribut commun (ligne, colonne ou diagonale) **gagne**.

### Environnements de test

En complément de Quarto, trois environnements de difficulté croissante servent à valider les algorithmes :

| Environnement | Type | Joueurs | Complexité |
|---------------|------|---------|------------|
| **LineWorld** | Navigation 1D | 1 | Minimale — valide que l'agent apprend à se déplacer |
| **GridWorld** | Navigation 2D | 1 | Faible — grille avec goal/fail, teste Q-Learning |
| **TicTacToe** | Jeu 2 joueurs | 2 | Moyenne — adversarial, state <= 27D |
| **Quarto** | Jeu 2 joueurs | 2 | Élevée — 16 pièces x 4 attributs, state 114D |

Chaque jeu 2 joueurs a une variante **VsRandom** (l'agent affronte un adversaire aléatoire) :
- `TicTacToeVsRandom` — agent = joueur X, adversaire = random
- `QuartoVsRandom` — agent = joueur 0, adversaire = random

---

## 2. Simulation random et parties/seconde

### Benchmark automatique

Au lancement de la GUI, un benchmark mesure la **vitesse de simulation** (500 parties random-vs-random) et affiche le résultat `parties/sec` dans le panneau d'informations.

```bash
# Démonstration console avec mesure parties/sec
python main.py --env lineworld       # RandomAgent sur LineWorld
python main.py --env gridworld       # Random vs Q-Learning sur GridWorld
python main.py --env tictactoe       # Random vs Random sur TicTacToe
python main.py --env quarto          # Random vs Random sur Quarto
```

### Méthode de mesure

```python
# GameViewer._benchmark_speed(n_games=500)
agent = RandomAgent(state_dim, n_actions)
start = time.time()
for _ in range(500):
    state = env.reset()
    while not env.is_game_over:
        action = agent.act(state, env.get_available_actions())
        state, _, _ = env.step(action)
gps = 500 / (time.time() - start)
```

Le résultat est affiché en **parties/seconde** dans le panneau gauche (Quarto) ou le panneau d'info (autres envs).

---

## 3. Jeu avec joueur humain + GUI

### 3 modes d'interaction

| Mode | Commande | Description |
|------|----------|-------------|
| **Observer** | `python main.py --gui --env quarto` | L'agent joue, l'humain regarde |
| **Humain vs IA** | `python main.py --play --env quarto` | L'humain joue contre un RandomAgent |
| **Humain vs Humain** | `python main.py --pvp --env quarto` | 2 joueurs sur le même écran |

### Interface graphique Quarto (layout 3 colonnes)

```
+----------------+---------------------+------------------+
|  Panneau       |                     |  Pièces          |
|  gauche        |    Plateau 4x4      |  disponibles     |
|                |   (indices 0-3)     |  (4x4 grille)    |
|  - Joueur      |                     |                  |
|  - Phase       |   Pièces 3D         |  Clic = give     |
|  - Aperçu      |   (cubes/cylindres) |                  |
|  - Boutons     |                     |                  |
|  - Stats       |   Clic = place      |                  |
|  - Résultat    |                     |                  |
+----------------+---------------------+------------------+
|  Barre de contrôle : raccourcis clavier + vitesse       |
+---------------------------------------------------------+
```

### Contrôles clavier (globaux)

| Touche | Action |
|--------|--------|
| **SPACE** | Pause / Reprendre |
| **N** | Avancer d'un pas |
| **R** | Restart la partie |
| **+/-** | Ajuster la vitesse |
| **F11** | Plein écran |
| **ESC** | Quitter |

### Contrôles par environnement (mode humain)

| Environnement | Entrée | Description |
|---------------|--------|-------------|
| LineWorld | ← / → | Gauche / Droite |
| GridWorld | ↑ / ↓ / ← / → | 4 directions |
| TicTacToe | 1-9 ou clic souris | Cases 1 à 9 |
| Quarto | 0-9, A-F ou clic souris | Position (place) ou pièce (give) |

### Fin de partie

À la fin d'une partie, l'affichage se met en **pause automatique** avec le message :
- « Joueur 0 gagne ! » / « Joueur 1 gagne ! » / « MATCH NUL »
- L'utilisateur appuie sur **R** (restart) ou **SPACE** (continuer)

---

## 4. Description de l'état du jeu (encoding)

### Quarto — Encodage compact (114 dimensions)

L'état est un vecteur `np.ndarray` de **114 dimensions**, décomposé en 4 blocs :

| Bloc | Dimensions | Description |
|------|-----------|-------------|
| Plateau | 80 | 16 positions x 5 canaux |
| Pièce courante | 16 | One-hot de la pièce à placer |
| Pièces disponibles | 16 | Masque binaire (1 = disponible) |
| Joueur courant | 2 | One-hot (joueur 0 ou 1) |
| **Total** | **114** | |

**Encodage par position du plateau (5 canaux) :**

```
[présence, tall, dark, solid, square]
```

- `présence = 1` si une pièce est posée, `0` sinon
- Les 4 attributs sont extraits des bits de l'ID de la pièce (0-15) :

```
tall   = piece_id & 1          # bit 0
dark   = (piece_id >> 1) & 1   # bit 1
solid  = (piece_id >> 2) & 1   # bit 2
square = (piece_id >> 3) & 1   # bit 3
```

**Exemple :** Pièce ID 13 (binaire `1101`) → tall=1, dark=0, solid=1, square=1


### Environnements de test

| Environnement | State dim | Encodage |
|---------------|-----------|----------|
| LineWorld(7) | 7 | One-hot de la position |
| GridWorld(5) | 25 | One-hot de (row, col) → `row * 5 + col` |
| TicTacToe | 27 | 9 cases x 3 canaux : `[vide, X, O]` |

---

## 5. Description d'une action du jeu (encoding)

### Quarto — Espace d'actions unifié (32 actions)

L'espace d'actions est un unique vecteur de taille 32, divisé en 2 blocs de sémantique fixe :

| Bloc | Indices | Phase | Sémantique |
|------|---------|-------|------------|
| **Place** | 0-15 | Placer | Position sur le plateau 4x4 (`row * 4 + col`) |
| **Give** | 16-31 | Donner | Pièce à donner (`action - 16` = ID pièce) |

**Masquage :** Pendant la phase "place", seules les actions 0-15 sont valides (cases vides). Pendant la phase "give", seules les actions 16-31 sont valides (pièces disponibles). Les actions interdites sont masquées par `get_available_actions()`.

```
action  0 → placer en position 0  (ligne 0, col 0)
action  1 → placer en position 1  (ligne 0, col 1)
...
action 15 → placer en position 15 (ligne 3, col 3)
action 16 → donner la pièce 0  (petit, clair, creux, rond)
action 17 → donner la pièce 1  (grand, clair, creux, rond)
...
action 31 → donner la pièce 15 (grand, foncé, plein, carré)
```

Ce design permet à un **unique réseau** de gérer les deux phases, avec masquage des actions interdites.

### Symétries (groupe D4)

La méthode `get_symmetries(state, action_probs)` retourne les **8 symétries** du plateau (4 rotations x 2 réflexions). Seuls les indices 0-15 (positions) sont permutés ; les indices 16-31 (pièces) restent inchangés.

### Environnements de test

| Environnement | n_actions | Sémantique |
|---------------|-----------|------------|
| LineWorld | 2 | 0 = Gauche, 1 = Droite |
| GridWorld | 4 | 0 = Haut, 1 = Bas, 2 = Gauche, 3 = Droite |
| TicTacToe | 9 | Position 0-8 sur la grille 3x3 |

---

## 6. Architecture du projet

```
deeprl/
├── deeprl/                    # Package Python
│   ├── __init__.py           # Exports : v0.2.0
│   ├── envs/                 # Environnements
│   │   ├── base.py           # Classe abstraite Environment
│   │   ├── line_world.py     # LineWorld (1D)
│   │   ├── grid_world.py     # GridWorld (2D)
│   │   ├── tictactoe.py      # TicTacToe + TicTacToeVsRandom
│   │   └── quarto.py         # Quarto + QuartoVsRandom
│   ├── agents/               # Agents
│   │   ├── base.py           # Classe abstraite Agent
│   │   ├── random_agent.py   # RandomAgent (baseline)
│   │   ├── human_agent.py    # HumanAgent (console/GUI)
│   │   └── tabular/
│   │       └── q_learning.py # TabularQLearning
│   ├── training/             # Entraînement
│   │   ├── trainer.py        # Boucle d'entraînement
│   │   ├── evaluator.py      # Évaluation + games_per_second
│   │   └── benchmark.py      # Comparaison d'agents
│   └── gui/
│       └── game_viewer.py    # GameViewer + HumanVsAgentViewer (Pygame)
├── main.py                   # Démos + GUI (--gui, --play, --pvp)
├── run_experiments.py         # Entraînement + métriques (JSON/PT/PNG)
├── SPECS.md                  # Ce document
├── COMMANDS.md               # Référence des commandes
├── requirements.txt
└── pyproject.toml
```

### Boucle d'interaction (MDP)

```
Agent                          Environment
  |                                |
  |-- act(state, available) ------>|
  |                                |-- step(action)
  |<-- (next_state, reward, done)--|
  |-- learn(s, a, r, s', done) -->|
```

### Conventions

| Élément | Type | Exemple |
|---------|------|---------|
| État | `np.ndarray` (1D aplati) | `shape=(114,)` |
| Action | `int` | `0` à `31` |
| Récompense | `float` | `+1.0`, `-1.0`, `0.0` |
| `training=True` | Exploration activée | epsilon-greedy |
| `training=False` | Exploitation pure | argmax Q |
| Jeux 2 joueurs | `current_player` ∈ {0, 1} | Alternance |

---

## 7. Interfaces abstraites

### Environment (`envs/base.py`)

```python
class Environment(ABC):
    # Propriétés abstraites
    state_shape -> Tuple[int, ...]   # Ex: (114,)
    n_actions   -> int               # Ex: 32

    # Propriétés dérivées
    state_dim      -> int            # prod(state_shape)
    current_player -> int            # 0 ou 1
    is_game_over   -> bool

    # Méthodes abstraites
    reset()                    -> np.ndarray
    step(action: int)          -> (np.ndarray, float, bool)
    get_available_actions()    -> List[int]

    # Méthodes fournies
    get_state()                -> np.ndarray   # copie
    is_action_valid(action)    -> bool
    render(mode="text")        -> Optional[str]
    clone()                    -> Environment  # deepcopy
```

### Agent (`agents/base.py`)

```python
class Agent(ABC):
    # Abstrait
    act(state, available_actions, training=True) -> int

    # Optionnels (défaut = no-op)
    learn(s, a, r, s', done)   -> Optional[Dict]
    on_episode_start()
    on_episode_end(total_reward, episode_length)
    save(path) / load(path)
    get_config()               -> Dict
```

---

## 8. Détails des agents implémentés

### RandomAgent — Baseline

| Propriété | Valeur |
|-----------|--------|
| Fichier | `agents/random_agent.py` |
| Apprentissage | Non |
| Comportement | Action uniforme parmi `available_actions` |

Sert à mesurer la performance de base et la vitesse de simulation.

### HumanAgent — Joueur interactif

| Propriété | Valeur |
|-----------|--------|
| Fichier | `agents/human_agent.py` |
| Modes | `"console"` (saisie texte) ou `"gui"` (Pygame) |
| Apprentissage | Non |

### TabularQLearning — Q-Learning tabulaire

| Propriété | Valeur |
|-----------|--------|
| Fichier | `agents/tabular/q_learning.py` |
| Algorithme | Q-Learning (Watkins, 1989) |
| Politique | epsilon-greedy avec décroissance exponentielle |

**Paramètres :**

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `lr` | 0.1 | Taux d'apprentissage α |
| `gamma` | 0.99 | Facteur de discount γ |
| `epsilon_start` | 1.0 | Exploration initiale |
| `epsilon_end` | 0.01 | Exploration minimale |
| `epsilon_decay` | 0.995 | Décroissance par épisode |

**Mise à jour (Bellman) :**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
