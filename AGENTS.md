# DeepRL — Specification Technique

Ce document constitue la **specification complete** de la bibliotheque **DeepRL**, une bibliotheque pedagogique de Deep Reinforcement Learning en PyTorch. Il decrit l'architecture, les interfaces, les environnements, les agents et les conventions de la version actuelle (branche `step1`).

---

## Table des matieres

1. [Structure du Projet](#structure-du-projet)
2. [Architecture Generale](#architecture-generale)
3. [Environnements](#environnements)
4. [Agents](#agents)
5. [Infrastructure d'Entrainement](#infrastructure-dentrainement)
6. [Interface Graphique](#interface-graphique)
7. [Interface en Ligne de Commande](#interface-en-ligne-de-commande)
8. [Encodage des Etats et Actions](#encodage-des-etats-et-actions)
9. [Dependances](#dependances)
10. [Guide du Developpeur](#guide-du-developpeur)
11. [References Academiques](#references-academiques)

---

## Structure du Projet

```
deeprl/
├── deeprl/                    # Package Python principal
│   ├── __init__.py           # Exports publics (version, classes, fonctions)
│   ├── envs/                 # Environnements
│   │   ├── base.py           # Classe abstraite Environment
│   │   ├── line_world.py     # LineWorld (navigation 1D)
│   │   ├── grid_world.py     # GridWorld (grille 2D avec etats terminaux)
│   │   ├── tictactoe.py      # TicTacToe / TicTacToeVsRandom (morpion)
│   │   └── quarto.py         # Quarto / QuartoVsRandom (jeu strategique)
│   │
│   ├── agents/               # Agents d'apprentissage par renforcement
│   │   ├── base.py           # Classe abstraite Agent
│   │   ├── random_agent.py   # Agent aleatoire (baseline)
│   │   ├── human_agent.py    # Agent humain (console ou GUI)
│   │   └── tabular/          # Methodes tabulaires
│   │       └── q_learning.py # Q-Learning avec table Q et epsilon-greedy
│   │
│   ├── training/             # Infrastructure d'entrainement et evaluation
│   │   ├── trainer.py        # Boucle d'entrainement (Trainer + TrainingMetrics)
│   │   ├── evaluator.py      # Evaluation (Evaluator + EvaluationResults)
│   │   └── benchmark.py      # Benchmarking comparatif (Benchmark + BenchmarkSuite)
│   │
│   └── gui/                  # Interface graphique
│       └── game_viewer.py    # Visualisation Pygame (GameViewer, HumanVsAgentViewer)
│
├── main.py                   # Demos interactives + GUI (observer / jouer)
├── run_experiments.py         # Entrainement complet + metriques (JSON/PT/PNG)
├── requirements.txt          # Dependances Python
├── pyproject.toml            # Configuration du package
└── AGENTS.md                 # Ce document (specification technique)
```

---

## Architecture Generale

La bibliotheque suit une architecture modulaire en couches :

```
┌─────────────────────────────────────────────────────┐
│              main.py / run_experiments.py            │  ← Points d'entree
├─────────────────────────────────────────────────────┤
│              training/ (Trainer, Evaluator)          │  ← Orchestration
├──────────────────────┬──────────────────────────────┤
│    agents/           │     envs/                     │  ← Logique metier
│  (act, learn)        │  (step, reset)                │
├──────────────────────┴──────────────────────────────┤
│              gui/ (GameViewer)                        │  ← Visualisation
└─────────────────────────────────────────────────────┘
```

**Separation des responsabilites :**
- `main.py` : **Execution et demonstration** — demos console, interface graphique (observer un agent, jouer en humain)
- `run_experiments.py` : **Entrainement et benchmarks** — sauvegarde modeles (.pt), metriques (JSON), graphiques (PNG)

**Boucle d'interaction standard** (MDP) :

```
Agent                          Environment
  │                                │
  │── act(state, available) ──────>│
  │                                │── step(action) ──>
  │<── (next_state, reward, done)──│
  │── learn(s, a, r, s', done) ──>│
  │                                │
```

**Conventions globales :**
- Les **etats** sont des `np.ndarray` (vecteurs 1D aplatis)
- Les **actions** sont des `int` (espace d'actions discret)
- Les **recompenses** sont des `float`
- `training=True` active l'exploration ; `training=False` exploite uniquement
- Les jeux a 2 joueurs utilisent `current_player` (0 ou 1) et `_winner`

---

## Environnements

### Interface abstraite (`envs/base.py`)

Tout environnement herite de `Environment` et implemente les methodes abstraites suivantes :

```python
class Environment(ABC):
    def __init__(self, name: str = "Environment"):
        self.name = name
        self._state: Optional[np.ndarray] = None
        self._done: bool = False
        self._current_player: int = 0

    # --- Proprietes abstraites ---
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Forme de l'espace d'etats. Ex: (25,) pour GridWorld 5x5."""

    @property
    def n_actions(self) -> int:
        """Nombre total d'actions dans l'espace d'actions discret."""

    # --- Proprietes derivees ---
    @property
    def state_dim(self) -> int:
        """Dimension aplatie = produit de state_shape."""

    @property
    def current_player(self) -> int:
        """Joueur courant (0 pour mono-joueur, 0/1 pour 2 joueurs)."""

    @property
    def is_game_over(self) -> bool:
        """True si l'episode est termine."""

    # --- Methodes abstraites ---
    def reset(self) -> np.ndarray:
        """Reinitialise l'environnement. Retourne l'etat initial."""

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute une action. Retourne (next_state, reward, done)."""

    def get_available_actions(self) -> List[int]:
        """Liste des actions valides dans l'etat courant."""

    # --- Methodes fournies par defaut ---
    def get_state(self) -> np.ndarray:
        """Copie de l'etat courant."""

    def is_action_valid(self, action: int) -> bool:
        """Verifie si une action est dans get_available_actions()."""

    def render(self, mode: str = "text") -> Optional[str]:
        """Affichage console ou image."""

    def clone(self) -> "Environment":
        """Copie profonde (deepcopy). Essentiel pour la planification."""
```

### Environnements implementes

#### 1. LineWorld — Navigation 1D

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `envs/line_world.py` |
| **Constructeur** | `LineWorld(size: int = 7)` |
| **state_shape** | `(size,)` — encodage one-hot de la position |
| **n_actions** | `2` — `0 = Gauche`, `1 = Droite` |
| **Position initiale** | `size // 2` (centre de la ligne) |
| **Etat terminal goal** | Position `size - 1` (extremite droite) → recompense `+1.0` |
| **Etat terminal fail** | Position `0` (extremite gauche) → recompense `-1.0` |
| **Recompense de pas** | `0.0` |
| **Joueurs** | 1 (mono-joueur) |

**Description :** Environnement minimal pour tester les algorithmes de base. L'agent demarre au centre d'une ligne de `size` cases et doit atteindre l'extremite droite (goal) tout en evitant l'extremite gauche (fail).

```python
from deeprl import LineWorld
env = LineWorld(size=7)  # 7 cases, agent au centre (position 3)
# state_shape = (7,), n_actions = 2, state_dim = 7
```

---

#### 2. GridWorld — Grille 2D avec etats terminaux

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `envs/grid_world.py` |
| **Constructeur** | `GridWorld(size: int = 5)` |
| **Methode de fabrique** | `GridWorld.create_simple(size: int = 5) -> GridWorld` |
| **state_shape** | `(size * size,)` — encodage one-hot de `(row, col)` |
| **n_actions** | `4` — `0 = Haut`, `1 = Bas`, `2 = Gauche`, `3 = Droite` |
| **Position initiale** | `(0, 0)` — coin superieur gauche |
| **Etat terminal goal** | `(size-1, size-1)` (coin inferieur droit) → recompense `+1.0` |
| **Etat terminal fail** | `(0, size-1)` (coin superieur droit) → recompense `-3.0` |
| **Recompense de pas** | `0.0` |
| **Joueurs** | 1 (mono-joueur) |

**Proprietes additionnelles :**
- `width`, `height` : dimensions de la grille
- `get_optimal_path_length()` : longueur du chemin optimal

```python
from deeprl import GridWorld
env = GridWorld.create_simple(size=5)  # 5x5, state_dim = 25
```

---

#### 3. TicTacToe — Morpion 2 joueurs

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `envs/tictactoe.py` |
| **Constructeur** | `TicTacToe(use_onehot: bool = True)` |
| **state_shape** | `(27,)` si `use_onehot=True` ; `(9,)` sinon |
| **n_actions** | `9` — positions 0 a 8 sur la grille 3x3 |
| **Encodage one-hot** | `[1,0,0]` = vide, `[0,1,0]` = X, `[0,0,1]` = O (par case) |
| **Encodage brut** | `0` = vide, `+1` = X, `-1` = O |
| **Joueurs** | 2 — `PLAYER_X = 0`, `PLAYER_O = 1` |
| **Recompense** | `+1.0` quand le joueur courant gagne, `0.0` sinon |

**Methode utilitaire :**
- `get_symmetries(state, action) -> List[Tuple[ndarray, int]]` : retourne les 8 symetries du plateau (rotations + reflexions)

##### Variante `TicTacToeVsRandom`

| Propriete | Valeur |
|-----------|--------|
| **Constructeur** | `TicTacToeVsRandom(use_onehot: bool = True, seed: Optional[int] = None)` |
| **Role de l'agent** | Toujours Player X (joueur 0) |
| **Adversaire** | Aleatoire (Player O joue uniformement) |
| **Recompenses** | `+1.0` (victoire), `-1.0` (defaite), `0.0` (nul) |

```python
from deeprl import TicTacToe, TicTacToeVsRandom
env = TicTacToe(use_onehot=True)        # 2 joueurs, state_dim = 27
env = TicTacToeVsRandom(use_onehot=True) # vs adversaire random
```

---

#### 4. Quarto — Jeu strategique 2 joueurs

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `envs/quarto.py` |
| **Constructeur** | `Quarto(use_compact_state: bool = True, seed: Optional[int] = None)` |
| **state_shape** | `(114,)` si `use_compact_state=True` ; `(306,)` sinon |
| **n_actions** | `32` — actions 0-15 = positions (place), actions 16-31 = pieces (give) |
| **Joueurs** | 2 (joueur 0 et joueur 1, alternance a chaque demi-tour) |
| **Recompense** | `+1.0` (victoire par alignement), `0.0` (match nul) |

**Description du jeu :** Quarto est un jeu de strategie sur un plateau 4x4 avec 16 pieces uniques ayant chacune 4 attributs binaires (tall/short, dark/light, solid/hollow, square/round). Le jeu alterne deux phases :

1. **Phase "give"** : Le joueur courant choisit une piece parmi les pieces disponibles et la *donne* a son adversaire. L'action est dans `[16, 31]` : action 16 = piece 0, ..., action 31 = piece 15.
2. **Phase "place"** : L'adversaire place la piece recue sur une case libre du plateau 4x4. L'action est dans `[0, 15]` representant la position sur le plateau.

Pendant la phase "place", les actions 16-31 sont masquees. Pendant la phase "give", les actions 0-15 sont masquees. Chaque sortie d'un reseau a toujours la meme semantique.

**Condition de victoire :** 4 pieces alignees (ligne, colonne ou diagonale) partageant au moins 1 attribut commun.

**Methode utilitaire :**
- `get_symmetries(state, action_probs) -> List[Tuple[ndarray, ndarray]]` : retourne les 8 symetries du plateau (groupe D4 : 4 rotations x 2 reflexions)

##### Variante `QuartoVsRandom`

| Propriete | Valeur |
|-----------|--------|
| **Constructeur** | `QuartoVsRandom(**kwargs)` |
| **Role de l'agent** | Joueur 0 |
| **Adversaire** | Aleatoire (joueur 1) |
| **Recompenses** | `+1.0` (victoire), `-1.0` (defaite), `0.0` (nul) |

```python
from deeprl import Quarto, QuartoVsRandom
env = Quarto(use_compact_state=True)       # 2 joueurs, state_dim = 114
env = QuartoVsRandom(use_compact_state=True) # vs adversaire random
```

---

### Tableau recapitulatif des environnements

| Environnement | Type | State Dim | Actions | Recompenses | Joueurs |
|---------------|------|-----------|---------|-------------|---------|
| `LineWorld(7)` | Navigation 1D | 7 | 2 (gauche/droite) | +1.0 / -1.0 / 0.0 | 1 |
| `GridWorld(5)` | Navigation 2D | 25 | 4 (haut/bas/gauche/droite) | +1.0 / -3.0 / 0.0 | 1 |
| `TicTacToe` | Jeu de plateau | 27 ou 9 | 9 (positions) | +1.0 / 0.0 | 2 |
| `TicTacToeVsRandom` | Jeu vs aleatoire | 27 ou 9 | 9 (positions) | +1.0 / -1.0 / 0.0 | 1 (vs bot) |
| `Quarto` | Jeu de strategie | 114 ou 306 | 32 (0-15 place + 16-31 give) | +1.0 / 0.0 | 2 |
| `QuartoVsRandom` | Jeu vs aleatoire | 114 ou 306 | 32 (0-15 place + 16-31 give) | +1.0 / -1.0 / 0.0 | 1 (vs bot) |

---

## Agents

### Interface abstraite (`agents/base.py`)

Tout agent herite de `Agent` et implemente au minimum `act()`.

```python
class Agent(ABC):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        name: str = "Agent",
        device: Optional[str] = None
    ):
        self.training_steps = 0
        self.episodes_played = 0

    @abstractmethod
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs
    ) -> int:
        """Choisit une action. Explore si training=True, exploite sinon."""

    def learn(self, state, action, reward, next_state, done, **kwargs) -> Optional[Dict]:
        """Met a jour l'agent. Retourne None par defaut (pas d'apprentissage)."""
        return None

    def on_episode_start(self) -> None:
        """Appelee au debut de chaque episode."""

    def on_episode_end(self, total_reward: float, episode_length: int) -> None:
        """Appelee a la fin de chaque episode."""

    def set_training_mode(self, training: bool) -> None:
        """Active/desactive le mode entrainement."""

    def save(self, path: str) -> None:
        """Sauvegarde l'agent (torch.save)."""

    def load(self, path: str) -> None:
        """Charge l'agent depuis le disque."""

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration (name, state_dim, n_actions, device)."""
```

### Agents implementes

---

#### 1. RandomAgent — Agent aleatoire (baseline)

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/random_agent.py` |
| **Categorie** | Baseline |
| **Apprentissage** | Non (`learn()` retourne `None`) |

```python
RandomAgent(
    state_dim: int,
    n_actions: int,
    seed: Optional[int] = None
)
```

**Comportement :** Choisit une action uniformement au hasard parmi `available_actions`. Sert de baseline pour evaluer les autres agents et pour mesurer la vitesse de simulation (parties/seconde).

---

#### 2. HumanAgent — Agent humain interactif

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/human_agent.py` |
| **Categorie** | Interactif |
| **Apprentissage** | Non |

```python
HumanAgent(
    state_dim: int = 0,
    n_actions: int = 0,
    mode: str = "console",   # "console" (saisie texte) ou "gui" (Pygame)
    name: str = "Human"
)
```

**Comportement :** Demande a l'utilisateur de choisir une action via le terminal ou l'interface Pygame.

---

#### 3. TabularQLearning — Q-Learning tabulaire

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/tabular/q_learning.py` |
| **Categorie** | Tabulaire |
| **Algorithme** | Q-Learning (Watkins, 1989) |
| **Politique** | epsilon-greedy avec decroissance exponentielle |

```python
TabularQLearning(
    n_states: int,
    n_actions: int,
    lr: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: Optional[int] = None
)
```

**Structure interne :**
- **Table Q** : `np.zeros((n_states, n_actions))`
- **Conversion d'etat** : Les etats one-hot sont convertis en indice via `argmax`

**Regle de mise a jour (Bellman) :**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Politique epsilon-greedy :**
- Avec probabilite epsilon : action aleatoire parmi `available_actions`
- Avec probabilite 1 - epsilon : `argmax Q(s, a)` parmi `available_actions`
- Decroissance : `epsilon <- max(epsilon_end, epsilon * epsilon_decay)` a chaque episode

```python
from deeprl import TabularQLearning
agent = TabularQLearning(n_states=25, n_actions=4, lr=0.1, gamma=0.99)
```

---

## Infrastructure d'Entrainement

### Trainer — Boucle d'entrainement

**Fichier :** `training/trainer.py`

```python
Trainer(
    env: Environment,
    agent: Agent,
    verbose: bool = True,
    log_interval: int = 100
)
```

```python
trainer.train(
    n_episodes: int,
    max_steps_per_episode: int = 1000,
    callbacks: Optional[List[Callable]] = None
) -> TrainingMetrics
```

**TrainingMetrics :**
- `get_summary(last_n=100) -> Dict` : retourne `mean_reward`, `std_reward`, `mean_length`, `max_reward`, `min_reward`

```python
from deeprl import Trainer
trainer = Trainer(env, agent, verbose=True, log_interval=100)
metrics = trainer.train(n_episodes=1000)
```

### Evaluator — Evaluation de performance

**Fichier :** `training/evaluator.py`

```python
Evaluator(
    env: Environment,
    agent: Agent,
    verbose: bool = True
)
```

```python
evaluator.evaluate(
    n_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    render: bool = False
) -> EvaluationResults
```

**EvaluationResults :**
- `get_summary() -> Dict` : retourne `mean_score`, `std_score`, `mean_length`, `std_length`, `mean_action_time`, `games_per_second`, et pour les jeux 2 joueurs : `win_rate`, `loss_rate`, `draw_rate`

**Metrique `games_per_second` :** Nombre de parties par seconde. Calculee comme `nombre_episodes / temps_total_secondes`.

```python
from deeprl import Evaluator
evaluator = Evaluator(env, agent, verbose=True)
results = evaluator.evaluate(n_episodes=100)
summary = results.get_summary()
print(f"Vitesse: {summary['games_per_second']:.1f} parties/sec")
```

### Benchmark — Comparaison d'agents

**Fichier :** `training/benchmark.py`

| Methode | Description |
|---------|-------------|
| `add_agent(name, agent_class, agent_kwargs=None)` | Ajoute un agent a comparer |
| `run() -> BenchmarkSuite` | Entraine et evalue tous les agents aux checkpoints |
| `quick_benchmark(env, agents, checkpoints, ...)` | Fonction utilitaire pour benchmark rapide |

---

## Interface Graphique

### GameViewer — Visualisation Pygame

**Fichier :** `gui/game_viewer.py`

```python
GameViewer(
    env: Environment,
    agent: Optional[Agent] = None,
    cell_size: int = 80,
    fps: int = 5,
    title: str = "DeepRL Viewer"
)
```

**Fonctionnalites :**
- Detection automatique du type d'environnement (LineWorld, GridWorld, TicTacToe, Quarto)
- Mode **agent** : l'agent joue automatiquement, l'humain observe
- Mode **humain** : l'humain joue via la souris ou le clavier
- **Benchmark de vitesse** : mesure automatiquement les `parties/seconde` au lancement (500 parties random) et l'affiche dans le panel d'info
- Suivi des statistiques : victoires, defaites, nuls
- Controles : SPACE (pause), N (step-by-step), fleches (vitesse), ESC (quitter)

### HumanVsAgentViewer — Mode Humain vs Agent

```python
HumanVsAgentViewer(
    env: Environment,
    opponent_agent: Agent,
    human_first: bool = True,
    cell_size: int = 80,
    fps: int = 30,
    title: str = "Humain vs Agent"
)
```

Sous-classe de `GameViewer` pour les jeux 2 joueurs. L'humain joue avec la souris/clavier, l'agent repond automatiquement.

### Fonctions publiques

```python
watch_agent(env, agent, n_episodes=10, fps=3)          # Observer un agent
play_human_vs_agent(env, agent, n_games=5, human_first=True)  # Jouer contre un agent
```

**Controles par environnement :**

| Environnement | Controles humain |
|---------------|-----------------|
| LineWorld | fleches gauche / droite |
| GridWorld | fleches haut / bas / gauche / droite |
| TicTacToe | Clic souris sur une case, ou touches 1-9 |
| Quarto (place) | Clic sur une case du plateau |
| Quarto (give) | Clic sur une piece dans le panel, ou touches 0-9/A-F |

---

## Interface en Ligne de Commande

### `main.py` — Demos et interface graphique

```bash
# Demos console (sortie terminal)
python main.py --env lineworld       # RandomAgent sur LineWorld + parties/sec
python main.py --env gridworld       # Random vs Q-Learning sur GridWorld + parties/sec
python main.py --env tictactoe       # Random vs Random sur TicTacToe + parties/sec
python main.py --env quarto          # Random vs Random sur Quarto + parties/sec

# Interface graphique — observer un agent
python main.py --gui --env tictactoe
python main.py --gui --env quarto

# Interface graphique — jouer en humain
python main.py --play --env tictactoe   # Humain vs RandomAgent
python main.py --play --env quarto      # Humain vs RandomAgent
python main.py --play --env gridworld   # Humain joue directement
python main.py --play --env lineworld   # Humain joue directement
```

| Argument | Description |
|----------|-------------|
| `--env` | Environnement : `lineworld`, `gridworld`, `tictactoe`, `quarto` (defaut: `gridworld`) |
| `--gui` | Lancer l'interface graphique pour observer un agent |
| `--play` | Jouer en humain contre un agent IA |
| `--agent` | Agent a utiliser : `Random`, `TabularQLearning` |

### `run_experiments.py` — Entrainement et benchmarks

```bash
python run_experiments.py                              # Tout lancer
python run_experiments.py --env gridworld               # Un seul env
python run_experiments.py --agent TabularQLearning      # Un seul agent
python run_experiments.py --checkpoints 1000,10000      # Checkpoints custom
python run_experiments.py --resume results/<run_dir>    # Reprendre un run
python run_experiments.py --plot results/<run_dir>      # Re-generer les plots
```

**Structure de sortie :**

```
results/<timestamp>/
├── config.json
├── summary.csv
├── <env>/
│   ├── metrics.json
│   ├── training_curves.json
│   ├── models/<AgentName>_ckpt<N>.pt
│   └── plots/ (bar_comparison.png, learning_curves.png, score_evolution.png)
```

---

## Encodage des Etats et Actions

Cette section decrit les choix d'encodage pour chaque environnement, necessaires pour les algorithmes d'apprentissage.

### LineWorld — Encodage one-hot (7D)

**Etat :** Vecteur one-hot de taille `size`. La position de l'agent est encodee par un `1` a l'indice correspondant.

```
Position 3 sur 7 cases -> [0, 0, 0, 1, 0, 0, 0]
```

**Action :** Entier 0 ou 1.
- `0` = Gauche
- `1` = Droite

### GridWorld — Encodage one-hot (25D)

**Etat :** Vecteur one-hot de taille `size x size`. La position `(row, col)` est convertie en indice `row * size + col`.

```
Position (2, 3) sur grille 5x5 -> indice 13 -> vecteur one-hot de taille 25
```

**Action :** Entier 0 a 3.
- `0` = Haut, `1` = Bas, `2` = Gauche, `3` = Droite

### TicTacToe — Encodage one-hot par case (27D)

**Etat (one-hot, 27D) :** Chaque case du plateau 3x3 est encodee par un vecteur de 3 valeurs :
- `[1, 0, 0]` = case vide
- `[0, 1, 0]` = X (joueur 0)
- `[0, 0, 1]` = O (joueur 1)

Les 9 cases sont concatenees -> vecteur de dimension 27.

**Etat (brut, 9D) :** `0` = vide, `+1` = X, `-1` = O.

**Action :** Entier 0 a 8 (position sur la grille 3x3, lecture gauche-droite, haut-bas).

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

### Quarto — Encodage compact (114D) et complet (306D)

#### Encodage compact (`use_compact_state=True`, 114 dimensions)

| Composante | Dimensions | Description |
|------------|-----------|-------------|
| Plateau | 80 | 16 positions x 5 canaux (presence + 4 attributs binaires) |
| Piece courante | 16 | Encodage one-hot de la piece a placer (0 si aucune) |
| Pieces disponibles | 16 | Masque binaire (1 = piece disponible) |
| Joueur courant | 2 | Encodage one-hot (joueur 0 ou joueur 1) |
| **TOTAL** | **114** | |

**Encodage par position du plateau (5 canaux) :**
```
[presence, tall, dark, solid, square]
```
- `presence = 1` si une piece est posee, `0` sinon
- `tall, dark, solid, square` : attributs binaires de la piece (extraits des bits de l'ID)

**Extraction des attributs depuis l'ID de piece (0-15) :**
```
tall   = bit 0 (piece_id & 1)
dark   = bit 1 (piece_id & 2) >> 1
solid  = bit 2 (piece_id & 4) >> 2
square = bit 3 (piece_id & 8) >> 3
```

**Exemple :** Piece ID 13 (binaire : 1101) -> tall=1, dark=0, solid=1, square=1

#### Encodage complet (`use_compact_state=False`, 306 dimensions)

| Composante | Dimensions | Description |
|------------|-----------|-------------|
| Plateau | 272 | 16 positions x 17 canaux (one-hot sur 16 pieces + indicateur vide) |
| Piece courante | 16 | Encodage one-hot |
| Pieces disponibles | 16 | Masque binaire |
| Joueur courant | 2 | Encodage one-hot |
| **TOTAL** | **306** | |

#### Actions Quarto

**Espace d'actions unifie :** 32 actions (entiers 0 a 31). L'espace est divise en deux blocs de semantique fixe :

| Bloc | Indices | Semantique | Actions valides |
|------|---------|------------|------------------|
| **Place** | 0-15 | Position sur le plateau 4x4 | Cases vides (masque pendant phase "give") |
| **Give** | 16-31 | Piece a donner (action 16 = piece 0, ..., 31 = piece 15) | Pieces disponibles (masque pendant phase "place") |

Pendant la phase "place", seules les actions 0-15 sont valides (cases vides).
Pendant la phase "give", seules les actions 16-31 sont valides (pieces disponibles).

**Correspondance sortie reseau → action :**
```
Sortie 0  : placer en position 0  (ligne 0, colonne 0)
Sortie 1  : placer en position 1  (ligne 0, colonne 1)
...
Sortie 15 : placer en position 15 (ligne 3, colonne 3)
Sortie 16 : donner la piece 0  (petit, clair, creux, rond)
Sortie 17 : donner la piece 1  (grand, clair, creux, rond)
...
Sortie 31 : donner la piece 15 (grand, fonce, plein, carre)
```

Ce design unifie permet a un seul reseau de gerer les deux phases, avec masquage des actions interdites de la phase non courante.

---

## Dependances

```
# Core
torch>=2.0.0
numpy>=1.24.0

# Visualisation
matplotlib>=3.7.0
pygame>=2.5.0

# Utilitaires
tqdm>=4.65.0
```

**Python requis :** >= 3.9

---

## Guide du Developpeur

### Ajouter un nouvel environnement

1. Creer `deeprl/envs/mon_env.py`
2. Heriter de `Environment` et implementer :
   - `state_shape` (propriete)
   - `n_actions` (propriete)
   - `reset()` -> `np.ndarray`
   - `step(action)` -> `(np.ndarray, float, bool)`
   - `get_available_actions()` -> `List[int]`
3. Exporter dans `deeprl/envs/__init__.py`
4. Exporter dans `deeprl/__init__.py`

### Ajouter un nouvel agent

1. Creer `deeprl/agents/<categorie>/mon_agent.py`
2. Heriter de `Agent` et implementer au minimum `act()`
3. Optionnellement, surcharger `learn()`, `save()`, `load()`
4. Exporter dans `deeprl/agents/__init__.py`
5. Exporter dans `deeprl/__init__.py`

### Conventions de code

| Convention | Description |
|------------|-------------|
| **Etats** | Toujours `np.ndarray` (vecteurs 1D aplatis) |
| **Actions** | Toujours `int` (espace discret) |
| **Recompenses** | Toujours `float` |
| **`training` flag** | `True` = exploration, `False` = exploitation |
| **Jeux 2 joueurs** | `current_player` (0 ou 1), `_winner` (None, 0, 1, ou -1 pour nul) |
| **Sauvegarde** | `torch.save` / `torch.load` (fichiers `.pt`) |
| **Nommage** | CamelCase pour les classes, snake_case pour les methodes |

---

## References Academiques

- **Q-Learning** : Watkins & Dayan, "Q-learning", Machine Learning (1992)
