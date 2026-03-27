# DeepRL — Specification Technique

Ce document constitue la **specification complete** de la bibliotheque **DeepRL**, une bibliotheque pedagogique de Deep Reinforcement Learning en PyTorch. Il decrit de maniere exhaustive l'architecture, les interfaces, les algorithmes, les parametres et les conventions de la bibliotheque.

---

## Table des matieres

1. [Structure du Projet](#structure-du-projet)
2. [Architecture Generale](#architecture-generale)
3. [Environnements](#environnements)
4. [Agents](#agents)
5. [Reseaux de Neurones](#reseaux-de-neurones)
6. [Memoire (Replay Buffers)](#memoire-replay-buffers)
7. [Infrastructure d'Entrainement](#infrastructure-dentrainement)
8. [Interface Graphique](#interface-graphique)
9. [Interface en Ligne de Commande](#interface-en-ligne-de-commande)
10. [Algorithmes — Synthese et References](#algorithmes--synthese-et-references)
11. [Dependances](#dependances)
12. [Guide du Developpeur](#guide-du-developpeur)
13. [Limitations Connues](#limitations-connues)
14. [References Academiques](#references-academiques)

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
│   │   ├── tabular/          # Methodes tabulaires
│   │   │   └── q_learning.py # Q-Learning avec table Q et epsilon-greedy
│   │   ├── value_based/      # Methodes basees sur la valeur
│   │   │   └── dqn.py        # DQN + Double DQN + Dueling DQN + PER
│   │   ├── policy_based/     # Methodes de gradient de politique
│   │   │   ├── reinforce.py  # REINFORCE + baselines (mean, critic)
│   │   │   └── ppo.py        # PPO avec GAE et masques d'actions
│   │   ├── planning/         # Methodes de planification
│   │   │   ├── mcts.py       # MCTS (UCT) + Random Rollout
│   │   │   ├── alphazero.py  # AlphaZero (Neural MCTS avec self-play)
│   │   │   ├── muzero.py     # MuZero (modele du monde appris)
│   │   │   └── muzero_stochastic.py  # MuZero Stochastique (noeuds de chance)
│   │   └── imitation/        # Apprentissage par imitation
│   │       └── expert_apprentice.py  # BC, DAgger, MCTSExpert, HumanExpert
│   │
│   ├── networks/             # Architectures de reseaux de neurones
│   │   └── mlp.py            # MLP, DuelingMLP, ActorCriticMLP
│   │
│   ├── memory/               # Buffers de replay
│   │   ├── replay_buffer.py  # ReplayBuffer (uniforme) + EpisodeBuffer
│   │   └── prioritized_buffer.py  # PrioritizedReplayBuffer (SumTree)
│   │
│   ├── training/             # Infrastructure d'entrainement et evaluation
│   │   ├── trainer.py        # Boucle d'entrainement (Trainer + TrainingMetrics)
│   │   ├── evaluator.py      # Evaluation (Evaluator + EvaluationResults)
│   │   └── benchmark.py      # Benchmarking comparatif (Benchmark + BenchmarkSuite)
│   │
│   └── gui/                  # Interface graphique
│       └── game_viewer.py    # Visualisation Pygame (GameViewer)
│
├── main.py                   # Demos interactives par environnement/algorithme
├── run_experiments.py         # Systeme d'experiences complet (local, JSON/PT/PNG)
├── requirements.txt          # Dependances Python
├── pyproject.toml            # Configuration du package
└── AGENTS.md                 # Ce document (specification technique)
```

---

## Architecture Generale

La bibliotheque suit une architecture modulaire en couches :

```
┌─────────────────────────────────────────────────────┐
│                    main.py / run_experiments.py      │  ← Point d'entree
├─────────────────────────────────────────────────────┤
│              training/ (Trainer, Evaluator)          │  ← Orchestration
├──────────────────────┬──────────────────────────────┤
│    agents/           │     envs/                     │  ← Logique metier
│  (act, learn)        │  (step, reset)                │
├──────────────────────┼──────────────────────────────┤
│  networks/ (MLP)     │  memory/ (ReplayBuffer)       │  ← Infrastructure
├──────────────────────┴──────────────────────────────┤
│              gui/ (GameViewer)                        │  ← Visualisation
└─────────────────────────────────────────────────────┘
```

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
- Le device PyTorch est auto-detecte (`"cuda"` si disponible, sinon `"cpu"`)

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
        self._current_player: int = 0  # Pour les jeux multi-joueurs

    # --- Proprietes abstraites (a implementer) ---
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Forme de l'espace d'etats. Ex: (25,) pour GridWorld 5x5."""

    @property
    def n_actions(self) -> int:
        """Nombre total d'actions dans l'espace d'actions discret."""

    # --- Proprietes derivees ---
    @property
    def state_dim(self) -> int:
        """Dimension aplatie = produit de state_shape. Utile pour les reseaux."""

    @property
    def current_player(self) -> int:
        """Joueur courant (0 pour mono-joueur, 0/1 pour 2 joueurs)."""

    @property
    def is_game_over(self) -> bool:
        """True si l'episode est termine."""

    # --- Methodes abstraites (a implementer) ---
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
        """Copie profonde (deepcopy). Essentiel pour MCTS/AlphaZero."""
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

**Description :** Environnement minimal pour tester les algorithmes de base. L'agent demarre au centre d'une ligne de `size` cases et doit atteindre l'extremite droite (goal) tout en evitant l'extremite gauche (fail). L'espace d'etats est un vecteur one-hot de taille `size`.

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
| **Obstacles / Murs** | Aucun (grille libre, l'agent est clipe aux bords) |

**Description :** Grille carree sans obstacles. L'agent part du coin superieur gauche et doit naviguer vers le coin inferieur droit (goal, `+1.0`) en evitant le coin superieur droit (fail, `-3.0`). Les deplacements hors de la grille sont ignores (l'agent reste sur place). Toutes les 4 actions sont toujours disponibles.

**Proprietes additionnelles :**
- `width`, `height` : dimensions de la grille
- `walls` : ensemble vide (pas de murs)
- `pos_to_index(pos)` : convertit `(row, col)` en indice d'etat
- `get_optimal_path_length()` : longueur du chemin optimal

```python
from deeprl import GridWorld
env = GridWorld(size=5)                # 5x5, state_dim = 25
env = GridWorld.create_simple(size=5)  # Equivalent
```

---

#### 3. TicTacToe — Morpion 2 joueurs

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `envs/tictactoe.py` |
| **Constructeur** | `TicTacToe(use_onehot: bool = True)` |
| **state_shape** | `(27,)` si `use_onehot=True` ; `(9,)` sinon |
| **n_actions** | `9` — positions 0 a 8 sur la grille 3×3 |
| **Encodage one-hot** | `[1,0,0]` = vide, `[0,1,0]` = X, `[0,0,1]` = O (par case) |
| **Encodage brut** | `0` = vide, `+1` = X, `-1` = O |
| **Joueurs** | 2 — `PLAYER_X = 0` (joue `+1`), `PLAYER_O = 1` (joue `-1`) |
| **Recompense (TicTacToe)** | `+1.0` quand le joueur courant gagne, `0.0` sinon |
| **Condition de victoire** | 3 symboles identiques en ligne, colonne ou diagonale |

**Methode utilitaire :**
- `get_symmetries(state, action) -> List[Tuple[ndarray, int]]` : retourne les 8 symetries du plateau (rotations + reflexions). Utile pour l'augmentation de donnees dans AlphaZero.

##### Variante `TicTacToeVsRandom`

| Propriete | Valeur |
|-----------|--------|
| **Constructeur** | `TicTacToeVsRandom(use_onehot: bool = True, seed: Optional[int] = None)` |
| **Role de l'agent** | Toujours Player X (joueur 0) |
| **Adversaire** | Aleatoire (Player O joue uniformement parmi les cases libres) |
| **Recompenses** | `+1.0` (victoire agent), `-1.0` (victoire adversaire), `0.0` (match nul) |

**Description :** Wrapper autour de `TicTacToe` qui simplifie l'entrainement en faisant jouer l'adversaire automatiquement de maniere aleatoire. L'agent voit uniquement ses propres tours. Si l'adversaire aleatoire gagne pendant son tour, l'agent recoit `-1.0`.

```python
from deeprl import TicTacToe, TicTacToeVsRandom

# Jeu 2 joueurs (pour MCTS, AlphaZero, self-play)
env = TicTacToe(use_onehot=True)  # state_dim = 27

# Entrainement contre adversaire aleatoire (pour DQN, PPO, REINFORCE)
env = TicTacToeVsRandom(use_onehot=True)  # state_dim = 27
```

---

#### 4. Quarto — Jeu strategique 2 joueurs

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `envs/quarto.py` |
| **Constructeur** | `Quarto(use_compact_state: bool = True, seed: Optional[int] = None)` |
| **state_shape** | `(114,)` si `use_compact_state=True` ; `(306,)` sinon |
| **n_actions** | `16` — indices de positions (phase "place") ou de pieces (phase "give") |
| **Joueurs** | 2 (joueur 0 et joueur 1, alternance a chaque demi-tour) |
| **Recompense** | `+1.0` (victoire par alignement), `0.0` (match nul) |

**Description du jeu :** Quarto est un jeu de strategie sur un plateau 4×4 avec 16 pieces uniques ayant chacune 4 attributs binaires (tall/short, dark/light, solid/hollow, square/round). Le jeu alterne deux phases :

1. **Phase "give"** : Le joueur courant choisit une piece parmi les pieces disponibles et la *donne* a son adversaire. L'action est un indice dans `[0, 15]` representant l'ID de la piece choisie.
2. **Phase "place"** : L'adversaire place la piece recue sur une case libre du plateau 4×4. L'action est un indice dans `[0, 15]` representant la position sur le plateau.

**Condition de victoire :** 4 pieces alignees (ligne, colonne ou diagonale) partageant au moins 1 attribut commun.

**Encodage d'etat compact (`114` dimensions) :**
- `16 × 5` = 80 : plateau (4 attributs + indicateur de presence par case)
- `16` : piece courante (encodage one-hot)
- `16` : pieces disponibles (masque binaire)
- `2` : joueur courant (one-hot)

**Encodage d'etat complet (`306` dimensions) :**
- `16 × 17` = 272 : plateau (16 IDs de pieces + indicateur vide, par case)
- `16` : piece courante
- `16` : pieces disponibles
- `2` : joueur courant

##### Variante `QuartoVsRandom`

| Propriete | Valeur |
|-----------|--------|
| **Constructeur** | `QuartoVsRandom(**kwargs)` |
| **Role de l'agent** | Joueur 0 |
| **Adversaire** | Aleatoire (joueur 1) — joue uniformement parmi les actions valides |
| **Recompenses** | `+1.0` (victoire agent), `-1.0` (victoire adversaire), `0.0` (match nul) |

```python
from deeprl import Quarto, QuartoVsRandom

# Jeu 2 joueurs (pour MCTS, AlphaZero)
env = Quarto(use_compact_state=True)  # state_dim = 114

# Entrainement contre adversaire aleatoire (pour DQN, PPO)
env = QuartoVsRandom(use_compact_state=True)  # state_dim = 114
```

---

### Tableau recapitulatif des environnements

| Environnement | Type | State Dim | Actions | Recompenses | Joueurs |
|---------------|------|-----------|---------|-------------|---------|
| `LineWorld(7)` | Navigation 1D | 7 | 2 (gauche/droite) | +1.0 / -1.0 / 0.0 | 1 |
| `GridWorld(5)` | Navigation 2D | 25 | 4 (haut/bas/gauche/droite) | +1.0 / -3.0 / 0.0 | 1 |
| `TicTacToe` | Jeu de plateau | 27 ou 9 | 9 (positions) | +1.0 / 0.0 | 2 |
| `TicTacToeVsRandom` | Jeu vs aleatoire | 27 ou 9 | 9 (positions) | +1.0 / -1.0 / 0.0 | 1 (vs bot) |
| `Quarto` | Jeu de strategie | 114 ou 306 | 16 (positions ou pieces) | +1.0 / 0.0 | 2 |
| `QuartoVsRandom` | Jeu vs aleatoire | 114 ou 306 | 16 (positions ou pieces) | +1.0 / -1.0 / 0.0 | 1 (vs bot) |

---

## Agents

### Interface abstraite (`agents/base.py`)

Tout agent herite de `Agent` et implemente au minimum `act()`. La methode `learn()` a une implementation par defaut qui retourne `None` (pas d'apprentissage).

```python
class Agent(ABC):
    def __init__(
        self,
        state_dim: int,       # Dimension de l'espace d'etats
        n_actions: int,        # Nombre d'actions possibles
        name: str = "Agent",   # Nom descriptif
        device: Optional[str] = None  # "cpu" ou "cuda" (auto-detecte si None)
    ):
        self.training_steps = 0
        self.episodes_played = 0

    # --- Methode abstraite (a implementer) ---
    @abstractmethod
    def act(
        self,
        state: np.ndarray,
        available_actions: Optional[List[int]] = None,
        training: bool = True,
        **kwargs  # Ex: env= pour MCTS/AlphaZero
    ) -> int:
        """Choisit une action. Explore si training=True, exploite sinon."""

    # --- Methodes avec implementation par defaut ---
    def learn(self, state, action, reward, next_state, done, **kwargs) -> Optional[Dict]:
        """Met a jour l'agent a partir d'une transition. Retourne None par defaut."""
        return None

    def on_episode_start(self) -> None:
        """Appelee au debut de chaque episode (reinitialisation interne)."""

    def on_episode_end(self, total_reward: float, episode_length: int) -> None:
        """Appelee a la fin de chaque episode. Incremente episodes_played."""

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

**Comportement :** Choisit une action uniformement au hasard parmi `available_actions`. Sert de baseline pour evaluer les autres agents.

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

**Comportement :** Demande a l'utilisateur de choisir une action via le terminal ou l'interface Pygame. Accepte un argument `env=` optionnel dans `act()` pour afficher le plateau avant la saisie.

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
    n_states: int,             # Nombre d'etats discrets
    n_actions: int,            # Nombre d'actions
    lr: float = 0.1,           # Taux d'apprentissage (alpha)
    gamma: float = 0.99,       # Facteur de discount
    epsilon_start: float = 1.0,     # Epsilon initial (exploration)
    epsilon_end: float = 0.01,      # Epsilon minimal
    epsilon_decay: float = 0.995,   # Facteur de decroissance par episode
    seed: Optional[int] = None
)
```

**Structure interne :**
- **Table Q** : `np.zeros((n_states, n_actions))` — stocke Q(s, a) pour chaque paire etat-action
- **Conversion d'etat** : Les etats one-hot sont convertis en indice via `argmax`

**Regle de mise a jour (Bellman) :**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Politique epsilon-greedy :**
- Avec probabilite epsilon : action aleatoire parmi `available_actions`
- Avec probabilite 1 - epsilon : `argmax Q(s, a)` parmi `available_actions`
- epsilon est decroissant : `epsilon ← max(epsilon_end, epsilon × epsilon_decay)` a chaque episode

```python
from deeprl import TabularQLearning
agent = TabularQLearning(n_states=25, n_actions=4, lr=0.1, gamma=0.99)
```

---

#### 4. DQNAgent — Deep Q-Network (+ variantes)

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/value_based/dqn.py` |
| **Categorie** | Value-based (reseau de neurones) |
| **Algorithme** | DQN (Mnih et al., 2013) avec extensions optionnelles |

```python
DQNAgent(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [64, 64],    # Couches cachees du MLP
    dueling: bool = False,                 # Architecture Dueling
    lr: float = 1e-3,                      # Taux d'apprentissage (Adam)
    gamma: float = 0.99,                   # Facteur de discount
    epsilon_start: float = 1.0,            # Exploration initiale
    epsilon_end: float = 0.01,             # Exploration minimale
    epsilon_decay: float = 0.995,          # Decroissance exponentielle
    use_replay: bool = True,               # Utiliser un replay buffer
    buffer_size: int = 10000,              # Taille du replay buffer
    batch_size: int = 64,                  # Taille des mini-batchs
    min_buffer_size: int = 1000,           # Taille minimale avant apprentissage
    prioritized: bool = False,             # Prioritized Experience Replay
    alpha: float = 0.6,                    # PER : exposant de priorite
    beta_start: float = 0.4,              # PER : biais d'importance initiale
    target_update_freq: int = 100,         # Frequence de MAJ du reseau cible
    soft_update: bool = False,             # MAJ douce (Polyak averaging)
    tau: float = 0.005,                    # Coefficient de MAJ douce
    double_dqn: bool = True,               # Double DQN (selection/evaluation decouples)
    device: Optional[str] = None,
    seed: Optional[int] = None
)
```

**Variantes combinables :**

| Variante | Parametre | Description |
|----------|-----------|-------------|
| **DQN standard** | (defauts) | Reseau Q + target network + experience replay |
| **Double DQN** | `double_dqn=True` | Decouple la selection d'action (Q-network) de l'evaluation (target). Reduit la surestimation des valeurs Q. |
| **Dueling DQN** | `dueling=True` | Separe Q(s,a) en V(s) + A(s,a) via `DuelingMLP`. Meilleure estimation pour les etats ou l'action importe peu. |
| **PER** | `prioritized=True` | Echantillonnage prioritaire proportionnel au TD-error. Necessite `use_replay=True`. |

**Architecture :**
- `q_network` : MLP ou DuelingMLP selon `dueling`
- `target_network` : Copie du q_network, mise a jour tous les `target_update_freq` pas (ou en continu via Polyak si `soft_update=True`)
- Gradient clipping : `max_norm = 10`

**Nommage automatique :** Le nom est construit dynamiquement — ex: `"Double DQN Dueling PER"` si toutes les variantes sont activees.

**Regle de mise a jour (Double DQN) :**

$$Q(s, a) \leftarrow r + \gamma \, Q_{\text{target}}(s', \arg\max_{a'} Q(s', a'))$$

```python
from deeprl import DQNAgent

# DQN minimal
agent = DQNAgent(state_dim=27, n_actions=9)

# DQN complet (toutes variantes)
agent = DQNAgent(
    state_dim=27, n_actions=9,
    hidden_dims=[128, 128],
    double_dqn=True, dueling=True, prioritized=True,
    buffer_size=10000, batch_size=64
)
```

---

#### 5. REINFORCEAgent — REINFORCE avec baselines

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/policy_based/reinforce.py` |
| **Categorie** | Policy Gradient |
| **Algorithme** | REINFORCE (Williams, 1992) |
| **Apprentissage** | Par episode (Monte-Carlo) |

```python
REINFORCEAgent(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [64, 64],    # Couches cachees
    lr: float = 1e-3,                      # Taux d'apprentissage (Adam)
    gamma: float = 0.99,                   # Facteur de discount
    baseline: str = "mean",                # Baseline : "none", "mean", "critic"
    critic_lr: float = 1e-3,              # LR du critique (si baseline="critic")
    entropy_coef: float = 0.01,           # Coefficient d'entropie (regularisation)
    device: Optional[str] = None,
    seed: Optional[int] = None
)
```

**Types de baseline :**

| Baseline | Description |
|----------|-------------|
| `"none"` | Pas de baseline. Variance elevee, convergence lente. |
| `"mean"` | Moyenne glissante des retours d'episodes (inter-episodes). Reduit la variance simplement. |
| `"critic"` | Reseau critique V(s) appris par MSE. Reduction optimale de la variance (Actor-Critic). Utilise `ActorCriticMLP`. |

**Fonctionnement :**
1. `act()` echantillonne une action selon la politique π(a|s) (softmax)
2. `learn()` stocke les transitions pendant l'episode
3. Quand `done=True`, `learn_from_episode()` calcule les retours discountes et met a jour :

**Gradient REINFORCE :**

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b_t)\right]$$

ou $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ et $b_t$ est la baseline.

**Regularisation par entropie :** Un bonus d'entropie est ajoute au loss pour encourager l'exploration :

$$\mathcal{L} = -\sum_t \log \pi(a_t|s_t)(G_t - b_t) - \beta \sum_t H(\pi(\cdot|s_t))$$

```python
from deeprl import REINFORCEAgent

# REINFORCE avec baseline mean
agent = REINFORCEAgent(state_dim=25, n_actions=4, baseline="mean")

# Actor-Critic (baseline critique)
agent = REINFORCEAgent(state_dim=25, n_actions=4, baseline="critic")
```

---

#### 6. PPOAgent — Proximal Policy Optimization

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/policy_based/ppo.py` |
| **Categorie** | Policy Gradient |
| **Algorithme** | PPO-Clip (Schulman et al., 2017) |
| **Architecture** | Actor-Critic avec couches partagees (optionnel) |

```python
PPOAgent(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [64, 64],    # Couches cachees
    shared_layers: bool = True,            # Couches partagees acteur/critique
    lr: float = 3e-4,                      # Taux d'apprentissage (Adam)
    gamma: float = 0.99,                   # Facteur de discount
    gae_lambda: float = 0.95,             # Lambda pour GAE
    clip_epsilon: float = 0.2,            # Clipping PPO
    n_epochs: int = 4,                     # Epochs par mise a jour
    batch_size: int = 64,                  # Taille des mini-batchs
    value_coef: float = 0.5,              # Poids du loss critique
    entropy_coef: float = 0.01,           # Poids du bonus d'entropie
    max_grad_norm: float = 0.5,           # Clipping du gradient (norme max)
    device: Optional[str] = None,
    seed: Optional[int] = None
)
```

**Architecture :**
- Reseau : `ActorCriticMLP` avec parametre `shared` pour partager ou separer les couches entre acteur et critique
- Buffer interne : `RolloutBuffer(n_actions)` qui stocke les transitions avec **masques d'actions** pour les espaces d'actions variables

**Generalized Advantage Estimation (GAE) :**

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

ou $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**Objectif PPO-Clip :**

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

ou $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$

**Masques d'actions :** PPO supporte les environnements ou les actions disponibles varient d'un etat a l'autre (ex: TicTacToe, Quarto). Les masques sont stockes dans le `RolloutBuffer` et reutilises lors des mises a jour pour eviter les log-prob sur des actions invalides.

**Methodes cles :**
- `act_with_value(state, available_actions, training) -> (action, log_prob, value)` : retourne l'action et les informations necessaires au stockage
- `store(state, action, reward, log_prob, value, done, action_mask=None)` : stocke une transition
- `update(last_value=0.0) -> Dict[str, float]` : met a jour le reseau (retourne loss/entropy/etc.)
- `learn()` : interface standard, stocke automatiquement et declenche `update()` quand `done=True` ou quand le buffer atteint une taille suffisante

```python
from deeprl import PPOAgent
agent = PPOAgent(
    state_dim=25, n_actions=4,
    hidden_dims=[64, 64],
    clip_epsilon=0.2, gae_lambda=0.95
)
```

---

#### 7. MCTSAgent — Monte Carlo Tree Search (UCT)

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/planning/mcts.py` |
| **Categorie** | Planification (sans apprentissage) |
| **Algorithme** | MCTS avec UCB1 (Kocsis & Szepesvari, 2006) |
| **Necessite `env=`** | Oui (pour cloner et simuler) |

```python
MCTSAgent(
    state_dim: int = 0,                    # Non utilise (planification pure)
    n_actions: int = 0,                    # Non utilise
    n_simulations: int = 100,              # Nombre de simulations MCTS
    c_exploration: float = math.sqrt(2),   # Constante d'exploration UCB1 (≈ 1.414)
    max_depth: int = 100,                  # Profondeur maximale des rollouts
    seed: Optional[int] = None
)
```

**Phases de MCTS :**
1. **Selection** : Descente dans l'arbre en suivant UCB1
2. **Expansion** : Creation d'un nouveau noeud enfant
3. **Simulation** : Rollout aleatoire jusqu'a un etat terminal
4. **Backpropagation** : Remontee de la valeur avec negation pour les jeux a 2 joueurs

**Formule UCB1 :**

$$\text{UCB1}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

**Backpropagation 2 joueurs :** La valeur est negee a chaque changement de joueur lors de la remontee, car une victoire pour un joueur est une defaite pour l'autre.

**Usage :** `act()` necessite l'argument `env=` pour cloner l'environnement et simuler des trajectoires. `learn()` retourne `None` (pas d'apprentissage — planification pure).

```python
from deeprl import MCTSAgent
mcts = MCTSAgent(n_simulations=100, c_exploration=1.41)
action = mcts.act(state, available_actions, env=env)  # env requis
```

---

#### 8. RandomRolloutAgent — Rollouts aleatoires

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/planning/mcts.py` |
| **Categorie** | Planification (sans apprentissage) |
| **Algorithme** | Evaluation par rollouts aleatoires |
| **Necessite `env=`** | Oui |

```python
RandomRolloutAgent(
    state_dim: int = 0,
    n_actions: int = 0,
    n_rollouts: int = 10,       # Nombre de rollouts par action candidate
    max_depth: int = 100,       # Profondeur maximale par rollout
    seed: Optional[int] = None
)
```

**Fonctionnement :** Pour chaque action candidate, execute `n_rollouts` rollouts aleatoires a partir de l'etat resultant et choisit l'action ayant la meilleure valeur moyenne. Plus simple que MCTS mais sans construction d'arbre.

```python
from deeprl import RandomRolloutAgent
rollout = RandomRolloutAgent(n_rollouts=20, max_depth=50)
action = rollout.act(state, available_actions, env=env)
```

---

#### 9. AlphaZeroAgent — AlphaZero (Neural MCTS)

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/planning/alphazero.py` |
| **Categorie** | Planification + Apprentissage |
| **Algorithme** | AlphaZero (Silver et al., 2017) |
| **Necessite `env=`** | Oui (pour MCTS dans l'espace reel) |

```python
AlphaZeroAgent(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [256, 256],   # Couches cachees
    n_simulations: int = 100,              # Simulations MCTS par decision
    c_puct: float = 1.0,                  # Constante d'exploration PUCT
    lr: float = 1e-3,                      # Taux d'apprentissage (Adam)
    weight_decay: float = 1e-4,           # Regularisation L2
    temperature: float = 1.0,             # Temperature pour selection d'action
    temperature_threshold: int = 15,       # Apres N coups, temperature → 0 (greedy)
    device: Optional[str] = None,
    seed: Optional[int] = None
)
```

**Architecture reseau (`AlphaZeroNetwork`) :**
- Tronc partage avec batch normalization
- **Tete politique** : MLP → softmax → π(a|s) ∈ [0, 1]^n_actions
- **Tete valeur** : MLP → tanh → V(s) ∈ [-1, 1]

**Formule PUCT (selection dans l'arbre) :**

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

ou P(s, a) est le prior donne par le reseau de politique.

**Workflow d'entrainement :**
1. **Self-play** : Generer des parties completes en utilisant MCTS guide par le reseau
2. **Collecte** : Stocker les triplets (etat, distribution MCTS π, resultat z)
3. **Entrainement** : Minimiser la loss combinee policy + value

$$\mathcal{L} = (z - V(s))^2 - \boldsymbol{\pi}^T \log \mathbf{p}(s)$$

**Methodes cles :**
- `self_play(env, n_games=100, verbose=False) -> List[Tuple[ndarray, ndarray, float]]` : genere `n_games` parties par self-play et retourne les exemples
- `train_on_examples(examples, batch_size=32, n_epochs=10) -> Dict[str, float]` : entraine le reseau sur les exemples collectes
- `get_action_probs(state, env, available_actions=None) -> Tuple[ndarray, float]` : retourne la distribution MCTS et la valeur estimee
- `learn()` : retourne `None` (l'entrainement se fait via le workflow self-play)

```python
from deeprl import TicTacToe, AlphaZeroAgent

env = TicTacToe()
agent = AlphaZeroAgent(state_dim=27, n_actions=9, n_simulations=50)

# Boucle self-play + entrainement
for iteration in range(10):
    examples = agent.self_play(env, n_games=100)
    metrics = agent.train_on_examples(examples, n_epochs=10)
    print(f"Iteration {iteration}: loss={metrics['total_loss']:.4f}")
```

---

#### 10. MuZeroAgent — MuZero (modele du monde appris)

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/planning/muzero.py` |
| **Categorie** | Model-based + Planification |
| **Algorithme** | MuZero (Schrittwieser et al., 2020) |
| **Necessite `env=`** | Non (planifie dans l'espace latent) |

```python
MuZeroAgent(
    state_dim: int,
    n_actions: int,
    latent_dim: int = 64,                  # Dimension de l'espace latent
    hidden_dims: List[int] = [128, 128],   # Couches cachees
    n_simulations: int = 50,               # Simulations MCTS par decision
    c_puct: float = 1.25,                 # Constante PUCT
    lr: float = 1e-3,                      # Taux d'apprentissage (Adam)
    gamma: float = 0.99,                   # Facteur de discount
    unroll_steps: int = 5,                 # Pas de deroulement pour l'entrainement
    buffer_size: int = 10000,              # Taille du replay buffer
    batch_size: int = 32,                  # Taille des mini-batchs
    device: Optional[str] = None,
    seed: Optional[int] = None
)
```

**Trois reseaux appris :**

| Reseau | Entree | Sortie | Role |
|--------|--------|--------|------|
| **Representation** h | observation o | etat latent s | Encoder l'observation reelle |
| **Dynamics** g | (etat latent s, action a) | (etat suivant s', recompense r) | Predire la dynamique dans l'espace latent |
| **Prediction** f | etat latent s | (politique π, valeur V) | Evaluer un etat latent |

**Principe :** Contrairement a AlphaZero qui planifie dans l'espace reel (en clonant l'environnement), MuZero apprend un modele du monde et planifie entierement dans l'espace latent. Cela lui permet de fonctionner dans des environnements ou le `clone()` n'est pas realiste.

**MCTS dans l'espace latent :** La recherche arborescente utilise le reseau de dynamics pour simuler les transitions et le reseau de prediction pour evaluer les noeuds.

**Entrainement (unrolling) :** A partir d'une trajectoire reelle, le modele deroule `unroll_steps` pas dans l'espace latent et minimise les erreurs de prediction de recompense, valeur et politique.

```python
from deeprl import MuZeroAgent

agent = MuZeroAgent(
    state_dim=25, n_actions=4,
    latent_dim=32, n_simulations=30
)

# MuZero apprend en continu via learn()
state = env.reset()
action = agent.act(state, env.get_available_actions(), training=True)
next_state, reward, done = env.step(action)
agent.learn(state, action, reward, next_state, done)
```

---

#### 11. StochasticMuZeroAgent — MuZero Stochastique

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/planning/muzero_stochastic.py` |
| **Categorie** | Model-based + Planification |
| **Algorithme** | Stochastic MuZero (Antonoglou et al., 2021) |

```python
StochasticMuZeroAgent(
    state_dim: int,
    n_actions: int,
    latent_dim: int = 64,                  # Dimension de l'espace latent
    chance_dim: int = 16,                  # Dimension des vecteurs de chance
    n_chance_outcomes: int = 32,           # Nombre de resultats stochastiques discrets
    hidden_dims: List[int] = [128, 128],   # Couches cachees
    n_simulations: int = 50,               # Simulations MCTS
    c_puct: float = 1.0,                  # Constante PUCT
    gamma: float = 0.99,                   # Facteur de discount
    lr: float = 1e-3,                      # Taux d'apprentissage
    unroll_steps: int = 5,                 # Pas de deroulement
    buffer_size: int = 10000,              # Taille du replay buffer
    batch_size: int = 32,                  # Taille des mini-batchs
    device: Optional[str] = None
)
```

**Extension de MuZero pour les environnements non-deterministes.** Le modele distingue les transitions deterministes (dues aux actions) des transitions stochastiques (dues a l'environnement).

**Cinq reseaux :**

| Reseau | Role |
|--------|------|
| **Representation** | Encode l'observation en etat latent |
| **Afterstate** | Applique une action → afterstate (partie deterministe) + recompense |
| **ChanceEncoder** | Encode la difference `(o_t, o_{t+1})` en chance outcome (Gumbel-Softmax) |
| **ChanceDynamics** | Applique un chance outcome a un afterstate → etat latent suivant |
| **Prediction** | Politique + valeur a partir d'un etat latent |
| **AfterstatePrediction** | Distribution de chance + valeur a partir d'un afterstate |

**Transition dans MuZero Stochastique :**

$$s_t \xrightarrow{\text{action } a} \bar{s}_t \xrightarrow{\text{chance } c} s_{t+1}$$

Le MCTS inclut des **noeuds de chance** en plus des noeuds de decision classiques.

```python
from deeprl import StochasticMuZeroAgent

agent = StochasticMuZeroAgent(
    state_dim=25, n_actions=4,
    latent_dim=32, n_chance_outcomes=16
)
```

---

#### 12. ExpertApprenticeAgent — Apprentissage par imitation

| Propriete | Valeur |
|-----------|--------|
| **Fichier** | `agents/imitation/expert_apprentice.py` |
| **Categorie** | Imitation Learning |
| **Algorithmes** | Behavior Cloning (BC), DAgger |

```python
ExpertApprenticeAgent(
    state_dim: int,
    n_actions: int,
    expert: ExpertPolicy,                  # Instance d'expert (MCTSExpert, HumanExpert)
    mode: str = "dagger",                  # "bc" ou "dagger"
    hidden_dims: List[int] = [64, 64],    # Couches cachees de l'apprenti
    lr: float = 1e-3,                      # Taux d'apprentissage
    device: Optional[str] = None,
    seed: Optional[int] = None
)
```

**Classes d'experts :**

```python
# Expert MCTS (planification)
MCTSExpert(
    n_simulations: int = 500,
    c_exploration: float = 1.41,
    seed: Optional[int] = None
)

# Expert humain (saisie console)
HumanExpert()
```

**Classes d'apprentis :**

```python
# Behavior Cloning : apprentissage supervise sur demonstrations fixes
BehaviorCloning(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [64, 64],
    lr: float = 1e-3,
    device: Optional[str] = None
)

# DAgger : aggregation de donnees avec correction par l'expert
DAgger(
    state_dim: int,
    n_actions: int,
    expert: ExpertPolicy,
    hidden_dims: List[int] = [64, 64],
    lr: float = 1e-3,
    beta_start: float = 1.0,    # Probabilite initiale de suivre l'expert
    beta_decay: float = 0.9,    # Decroissance du beta a chaque iteration
    device: Optional[str] = None
)
```

**Behavior Cloning (BC) :**
1. L'expert genere des demonstrations : paires (etat, action_expert)
2. L'apprenti apprend par classification supervisee sur ces paires
3. Probleme : distribution shift — l'apprenti rencontre des etats non vus pendant le test

**DAgger (Dataset Aggregation, Ross et al., 2011) :**
1. L'apprenti joue dans l'environnement (potentiellement en mixant expert/apprenti)
2. L'expert etiquette les etats rencontres par l'apprenti
3. Les nouvelles donnees sont ajoutees au dataset et l'apprenti est re-entraine
4. `beta` (probabilite de suivre l'expert) decroit au fil des iterations

**Methodes cles de `ExpertApprenticeAgent` :**
- `collect_demonstrations(env, n_episodes=100, verbose=False)` : collecte des demos (BC uniquement)
- `train(env, n_iterations=10, episodes_per_iteration=20, epochs_per_iteration=50, verbose=True) -> List[Dict]` : boucle d'entrainement complete

```python
from deeprl import ExpertApprenticeAgent, MCTSExpert, TicTacToe

env = TicTacToe()
expert = MCTSExpert(n_simulations=500)

# Behavior Cloning
bc_agent = ExpertApprenticeAgent(
    state_dim=27, n_actions=9,
    expert=expert, mode="bc"
)
bc_agent.collect_demonstrations(env, n_episodes=100)
bc_agent.train(env, n_iterations=5, epochs_per_iteration=50)

# DAgger (correction interactive)
dagger_agent = ExpertApprenticeAgent(
    state_dim=27, n_actions=9,
    expert=expert, mode="dagger"
)
dagger_agent.train(env, n_iterations=10, episodes_per_iteration=20)
```

---

## Reseaux de Neurones

### Fichier : `networks/mlp.py`

Trois architectures sont fournies, toutes basees sur des perceptrons multi-couches.

#### MLP — Perceptron multi-couches generique

```python
MLP(
    state_dim: int,                            # Dimension d'entree
    output_dim: int,                           # Dimension de sortie
    hidden_dims: List[int] = [64, 64],        # Tailles des couches cachees
    activation: str = "relu",                 # Activation interne
    dropout: float = 0.0,                     # Taux de dropout
    output_activation: Optional[str] = None   # Activation de sortie
)
```

**Activations supportees :** `"relu"`, `"tanh"`, `"elu"`, `"leaky_relu"`, `"silu"`
**Activations de sortie :** `"softmax"`, `"tanh"`, `"sigmoid"`, `None` (lineaire)
**Initialisation :** Xavier uniform (poids), zeros (biais)

Utilise par : `DQNAgent` (sans dueling), `REINFORCEAgent` (sans critic), `BehaviorCloning`, `DAgger`.

#### DuelingMLP — Architecture Dueling

```python
DuelingMLP(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [64, 64],
    activation: str = "relu"
)
```

**Decomposition :**

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

Le tronc commun (`feature_layer`) est suivi de deux branches separees :
- **Value stream** → V(s) scalaire
- **Advantage stream** → A(s, a) vecteur de taille n_actions

**Methodes :** `get_value(x)`, `get_advantage(x)`

Utilise par : `DQNAgent` (quand `dueling=True`).

#### ActorCriticMLP — Architecture Acteur-Critique

```python
ActorCriticMLP(
    state_dim: int,
    n_actions: int,
    hidden_dims: List[int] = [64, 64],
    activation: str = "relu",
    shared: bool = True   # Couches partagees entre acteur et critique
)
```

**Sorties :** `forward(x) -> (policy_logits, value)`
- **policy_logits** : vecteur de taille n_actions (avant softmax)
- **value** : scalaire V(s)

**Mode `shared=True`** : Tronc commun → tete politique + tete valeur
**Mode `shared=False`** : Deux reseaux completement separes

**Methodes :** `get_policy(x)`, `get_value(x)`, `get_action_and_value(x, action)`

Utilise par : `REINFORCEAgent` (quand `baseline="critic"`), `PPOAgent`.

---

## Memoire (Replay Buffers)

### ReplayBuffer — Buffer circulaire uniforme

**Fichier :** `memory/replay_buffer.py`

```python
ReplayBuffer(capacity: int = 10000)
```

Stocke des transitions (s, a, r, s', done) dans un buffer circulaire (deque). L'echantillonnage est uniforme.

| Methode | Description |
|---------|-------------|
| `push(state, action, reward, next_state, done)` | Ajoute une transition |
| `sample(batch_size) -> Tuple` | Echantillonne un batch aleatoire (states, actions, rewards, next_states, dones) |
| `is_ready(batch_size) -> bool` | True si le buffer contient assez de transitions |
| `clear()` | Vide le buffer |
| `get_stats() -> Dict` | Statistiques (taille, capacite, etc.) |

Utilise par : `DQNAgent` (quand `prioritized=False`).

### EpisodeBuffer — Buffer d'episode

**Fichier :** `memory/replay_buffer.py`

```python
EpisodeBuffer()
```

Stocke les transitions d'un seul episode pour les methodes Monte-Carlo.

| Methode | Description |
|---------|-------------|
| `push(state, action, reward, log_prob=None, value=None)` | Ajoute une transition |
| `get_returns(gamma=0.99) -> ndarray` | Calcule les retours discountes $G_t$ |
| `get_advantages(gamma=0.99) -> ndarray` | Calcule les avantages centres-reduits |
| `get_batch() -> Tuple` | Retourne toutes les donnees de l'episode |
| `clear()` | Reinitialise pour un nouvel episode |

Utilise par : `REINFORCEAgent`.

### PrioritizedReplayBuffer — Buffer avec echantillonnage prioritaire

**Fichier :** `memory/prioritized_buffer.py`

```python
PrioritizedReplayBuffer(
    capacity: int = 10000,
    alpha: float = 0.6,         # Exposant de priorite (0 = uniforme, 1 = pure priorite)
    beta_start: float = 0.4,    # Correction de biais initiale
    beta_end: float = 1.0,      # Correction de biais finale
    beta_frames: int = 100000,  # Nombre de frames pour l'annealing de beta
    epsilon: float = 1e-6       # Petite constante pour eviter priorite nulle
)
```

**Implementation :** Utilise un `SumTree` pour un echantillonnage proportionnel en O(log n). Les nouvelles transitions recoivent `max_priority` pour garantir au moins un echantillonnage.

| Methode | Description |
|---------|-------------|
| `push(state, action, reward, next_state, done)` | Ajoute avec priorite maximale |
| `sample(batch_size, beta=None) -> Tuple` | Retourne (batch, importance_weights, indices) |
| `update_priorities(indices, td_errors)` | Met a jour les priorites apres calcul TD |
| `is_ready(batch_size) -> bool` | True si assez de transitions |

**Principe (Schaul et al., 2015) :** Les transitions avec un grand TD-error sont echantillonnees plus souvent. Les poids d'importance-sampling corrigent le biais introduit.

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

Utilise par : `DQNAgent` (quand `prioritized=True`).

---

## Infrastructure d'Entrainement

### Trainer — Boucle d'entrainement

**Fichier :** `training/trainer.py`

```python
Trainer(
    env: Environment,
    agent: Agent,
    verbose: bool = True,
    log_interval: int = 100    # Frequence d'affichage (en episodes)
)
```

```python
trainer.train(
    n_episodes: int,
    max_steps_per_episode: int = 1000,
    callbacks: Optional[List[Callable]] = None
) -> TrainingMetrics
```

**Boucle d'entrainement :**
1. `agent.on_episode_start()`
2. `state = env.reset()`
3. Boucle de pas :
   - `action = agent.act(state, env.get_available_actions(), training=True, env=env)`
   - `next_state, reward, done = env.step(action)`
   - `agent.learn(state, action, reward, next_state, done)`
4. `agent.on_episode_end(total_reward, episode_length)`

**Note :** Le `Trainer` passe `env=self.env` a `agent.act()`, ce qui permet aux agents MCTS/AlphaZero de cloner l'environnement pour leurs simulations.

**TrainingMetrics :** Stocke les recompenses et longueurs par episode.
- `get_summary(last_n=100) -> Dict` : retourne `mean_reward`, `std_reward`, `mean_length`, `max_reward`, `min_reward` sur les N derniers episodes

```python
from deeprl import Trainer

trainer = Trainer(env, agent, verbose=True, log_interval=100)
metrics = trainer.train(n_episodes=1000)
summary = metrics.get_summary()
print(f"Reward moyen: {summary['mean_reward']:.2f}")
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

**Fonctionnement :**
1. Active le mode evaluation : `agent.set_training_mode(False)`
2. Joue `n_episodes` episodes complets
3. Restaure le mode entrainement apres evaluation

**EvaluationResults :**
- `get_summary() -> Dict` : retourne `mean_score`, `std_score`, `mean_length`, `mean_action_time`, `win_rate`, `loss_rate`, `draw_rate` (ces derniers pour les jeux a 2 joueurs uniquement)

```python
from deeprl import Evaluator

evaluator = Evaluator(env, agent, verbose=True)
results = evaluator.evaluate(n_episodes=100)
summary = results.get_summary()
print(f"Score: {summary['mean_score']:.2f} +/- {summary['std_score']:.2f}")
print(f"Win rate: {summary['win_rate']:.1%}")  # Pour les jeux 2 joueurs
```

### Benchmark — Comparaison d'agents

**Fichier :** `training/benchmark.py`

```python
Benchmark(
    env: Environment,
    checkpoints: List[int] = None,       # Default: [1000, 10000, 100000]
    eval_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    log_interval: int = 100,
    verbose: bool = True
)
```

| Methode | Description |
|---------|-------------|
| `add_agent(name, agent_class, agent_kwargs=None)` | Ajoute un agent a comparer |
| `run() -> BenchmarkSuite` | Entraine et evalue tous les agents aux checkpoints |

**BenchmarkSuite :**
- `get_comparison_table() -> str` : Table de comparaison formatee
- `get_csv_data() -> str` : Export CSV des resultats

```python
from deeprl.training import Benchmark

bench = Benchmark(env, checkpoints=[1000, 10000])
bench.add_agent('Random', RandomAgent, {'state_dim': 25, 'n_actions': 4})
bench.add_agent('Q-Learning', TabularQLearning, {'n_states': 25, 'n_actions': 4})
suite = bench.run()
print(suite.get_comparison_table())
```

---

## Interface Graphique

### GameViewer — Visualisation Pygame

**Fichier :** `gui/game_viewer.py`

```python
GameViewer(
    env: Environment,
    agent: Optional[Agent] = None,  # None = mode humain interactif
    cell_size: int = 80,
    fps: int = 5,
    title: str = "DeepRL Viewer"
)
```

```python
viewer.run(n_episodes: Optional[int] = None)  # None = infini
```

**Fonctionnalites :**
- Detection automatique du type d'environnement via `env.name` (LineWorld, GridWorld, TicTacToe, Quarto)
- Mode **agent** : l'agent joue automatiquement, l'humain observe
- Mode **humain** : l'humain joue via la souris ou le clavier
- Suivi des statistiques : `wins`, `losses`, `draws`, `episode_count`

**Prerequis :** `pygame >= 2.5.0` (optionnel — le package fonctionne sans pour l'entrainement)

---

## Interface en Ligne de Commande

### `main.py` — Demos interactives

```bash
# Demos par environnement/algorithme
python main.py --env lineworld       # Q-Learning sur LineWorld
python main.py --env gridworld       # Q-Learning sur GridWorld
python main.py --env tictactoe       # DQN sur TicTacToe
python main.py --env reinforce       # REINFORCE sur GridWorld
python main.py --env ppo             # PPO sur GridWorld
python main.py --env mcts            # MCTS sur TicTacToe
python main.py --env alphazero       # AlphaZero sur TicTacToe
python main.py --env muzero          # MuZero sur GridWorld
python main.py --env stochastic-muzero  # MuZero Stochastique sur GridWorld
python main.py --env quarto          # MCTS sur Quarto
python main.py --env imitation       # BC/DAgger sur TicTacToe

# Autres modes
python main.py --benchmark           # Benchmark comparatif avec graphiques
python main.py --gui                 # Interface graphique Pygame
python main.py --play                # Jouer en tant qu'humain
```

### `run_experiments.py` — Systeme d'experiences complet

Systeme d'experiences local generant des resultats au format JSON/PT/PNG, avec support de reprise et filtrage.

```bash
# Lancer toutes les experiences
python run_experiments.py

# Filtrer par environnement et/ou agent
python run_experiments.py --env gridworld
python run_experiments.py --agent TabularQLearning
python run_experiments.py --env tictactoe --agent DQN

# Checkpoints personnalises
python run_experiments.py --checkpoints 1000,10000,100000

# Reprendre une experience interrompue
python run_experiments.py --resume results/<run_dir>

# Regenerer les graphiques
python run_experiments.py --plot results/<run_dir>

# Evaluer uniquement (sans re-entrainer)
python run_experiments.py --eval results/<run_dir>
```

**Structure de sortie :**

```
results/<timestamp>/
├── summary.csv                          # Resume global (tous envs × agents)
├── <env>/
│   ├── run_config.json                  # Configuration de l'experience
│   ├── metrics.json                     # Metriques d'evaluation par agent et checkpoint
│   ├── training_curves.json             # Courbes de recompense episode par episode
│   ├── models/
│   │   └── <AgentName>_<checkpoint>.pt  # Modeles sauvegardes
│   └── plots/
│       ├── bar_comparison.png           # Barres comparatives par checkpoint
│       ├── learning_curves.png          # Courbes d'apprentissage
│       └── score_evolution.png          # Evolution du score moyen
```

---

## Algorithmes — Synthese et References

### Value-Based (estimation de Q(s, a))

| Algorithme | Idee cle | Fichier |
|------------|----------|---------|
| **Q-Learning** | Table Q + Bellman update + epsilon-greedy | `agents/tabular/q_learning.py` |
| **DQN** | Reseau Q + target network + experience replay | `agents/value_based/dqn.py` |
| **Double DQN** | Decouple selection et evaluation pour reduire la surestimation | `agents/value_based/dqn.py` |
| **Dueling DQN** | Decompose Q(s,a) = V(s) + A(s,a) | `agents/value_based/dqn.py` |
| **PER** | Echantillonnage prioritaire selon le TD-error | `memory/prioritized_buffer.py` |

### Policy Gradient (optimisation directe de π(a|s))

| Algorithme | Idee cle | Fichier |
|------------|----------|---------|
| **REINFORCE** | Gradient Monte-Carlo sur la politique | `agents/policy_based/reinforce.py` |
| **REINFORCE + Baseline** | Soustraction d'une baseline pour reduire la variance | `agents/policy_based/reinforce.py` |
| **PPO** | Clipping du ratio de politique + GAE + entropy bonus | `agents/policy_based/ppo.py` |

### Planning / Model-Based

| Algorithme | Idee cle | Fichier |
|------------|----------|---------|
| **MCTS (UCT)** | Recherche arborescente avec UCB1. Planification pure, pas d'apprentissage. | `agents/planning/mcts.py` |
| **Random Rollout** | Evaluation par rollouts aleatoires (sans arbre) | `agents/planning/mcts.py` |
| **AlphaZero** | MCTS guide par un reseau (politique + valeur) entraine par self-play | `agents/planning/alphazero.py` |
| **MuZero** | Comme AlphaZero mais avec un modele du monde appris (3 reseaux). Planifie dans l'espace latent. | `agents/planning/muzero.py` |
| **MuZero Stochastique** | Extension de MuZero avec des noeuds de chance pour les transitions stochastiques (5 reseaux). | `agents/planning/muzero_stochastic.py` |

### Imitation Learning

| Algorithme | Idee cle | Fichier |
|------------|----------|---------|
| **Behavior Cloning** | Apprentissage supervise sur les demonstrations de l'expert | `agents/imitation/expert_apprentice.py` |
| **DAgger** | Aggregation de donnees : l'expert corrige les etats visites par l'apprenti | `agents/imitation/expert_apprentice.py` |

---

## Dependances

```
# Deep Learning
torch>=2.0.0
numpy>=1.24.0

# Visualisation
matplotlib>=3.7.0
pygame>=2.5.0        # Optionnel (necessaire uniquement pour le GUI)

# Utilitaires
tqdm>=4.65.0
```

---

## Guide du Developpeur

### Ajouter un nouvel environnement

1. Creer `deeprl/envs/mon_env.py`
2. Heriter de `Environment` et implementer :
   - `state_shape` (propriete)
   - `n_actions` (propriete)
   - `reset()` → `np.ndarray`
   - `step(action)` → `(np.ndarray, float, bool)`
   - `get_available_actions()` → `List[int]`
3. Exporter dans `deeprl/envs/__init__.py`
4. Exporter dans `deeprl/__init__.py`

### Ajouter un nouvel agent

1. Creer `deeprl/agents/<categorie>/mon_agent.py`
2. Heriter de `Agent` et implementer au minimum `act()`
3. Optionnellement, surcharger `learn()`, `save()`, `load()`, `on_episode_start()`, `on_episode_end()`
4. Exporter dans `deeprl/agents/<categorie>/__init__.py`
5. Exporter dans `deeprl/agents/__init__.py`
6. Exporter dans `deeprl/__init__.py`

### Conventions de code

| Convention | Description |
|------------|-------------|
| **Etats** | Toujours `np.ndarray` (vecteurs 1D aplatis) |
| **Actions** | Toujours `int` (espace discret) |
| **Recompenses** | Toujours `float` |
| **`training` flag** | `True` = exploration active, `False` = exploitation pure |
| **Jeux 2 joueurs** | `current_player` (0 ou 1), `_winner` (None, 0, 1, ou -1 pour nul) |
| **Device PyTorch** | Auto-detecte si non specifie (`"cuda"` si disponible) |
| **Sauvegarde** | `torch.save` / `torch.load` (fichiers `.pt`) |
| **Nommage** | CamelCase pour les classes, snake_case pour les methodes |

---

## Limitations Connues

1. **DQN target network** : Le reseau cible ne masque pas les actions invalides dans `max Q(s', a')`. En pratique, cela n'affecte pas la convergence car les actions invalides sont rarement selectionnees, mais ce n'est pas strictement optimal.

2. **MuZero available_actions** : MuZero utilise les `available_actions` du noeud racine pour tous les noeuds de l'arbre. C'est conforme au papier original (dans l'espace latent, les actions valides ne sont pas connues), mais peut etre imprecis pour les jeux ou les actions changent drastiquement.

---

## References Academiques

- **Q-Learning** : Watkins & Dayan, "Q-learning", Machine Learning (1992)
- **DQN** : Mnih et al., "Playing Atari with Deep Reinforcement Learning", NIPS Workshop (2013)
- **Double DQN** : van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI (2016)
- **Dueling DQN** : Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML (2016)
- **PER** : Schaul et al., "Prioritized Experience Replay", ICLR (2016)
- **REINFORCE** : Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", Machine Learning (1992)
- **PPO** : Schulman et al., "Proximal Policy Optimization Algorithms", arXiv (2017)
- **GAE** : Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR (2016)
- **MCTS** : Kocsis & Szepesvari, "Bandit based Monte-Carlo Planning", ECML (2006)
- **AlphaZero** : Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv (2017)
- **MuZero** : Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", Nature (2020)
- **MuZero Stochastique** : Antonoglou et al., "Planning in Stochastic Environments with a Learned Model", ICLR (2022)
- **Behavior Cloning** : Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network", NeurIPS (1989)
- **DAgger** : Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS (2011)