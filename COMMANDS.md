# DeepRL — Commandes disponibles

> Voir [SPECS.md](SPECS.md) pour la spécification technique complète (encoding, architecture, interfaces).

## Démos console

```bash
python main.py --env lineworld       # RandomAgent sur LineWorld + parties/sec
python main.py --env gridworld       # Random vs Q-Learning sur GridWorld + parties/sec
python main.py --env tictactoe       # Random vs Random sur TicTacToe + parties/sec
python main.py --env quarto          # Random vs Random sur Quarto + parties/sec
```

## Interface graphique — Observer un agent

```bash
python main.py --gui --env lineworld                  # Observer un agent sur LineWorld
python main.py --gui --env gridworld                  # Observer un agent sur GridWorld
python main.py --gui --env tictactoe                  # Observer un agent sur TicTacToe
python main.py --gui --env quarto                     # Observer un agent sur Quarto
python main.py --gui --env quarto --agent Random      # Forcer un agent spécifique
```

## Jouer contre l'IA (Humain vs Agent)

```bash
python main.py --play --env lineworld    # Jouer seul sur LineWorld (clavier)
python main.py --play --env gridworld    # Jouer seul sur GridWorld (clavier)
python main.py --play --env tictactoe    # Humain vs IA sur TicTacToe
python main.py --play --env quarto       # Humain vs IA sur Quarto
```

## Humain vs Humain (même écran)

```bash
python main.py --pvp --env tictactoe    # 2 joueurs humains sur TicTacToe
python main.py --pvp --env quarto       # 2 joueurs humains sur Quarto
```

## Entraînement et benchmarks

```bash
python run_experiments.py                              # Lancer tous les entraînements
python run_experiments.py --env gridworld               # Un seul environnement
python run_experiments.py --agent TabularQLearning      # Un seul agent
python run_experiments.py --checkpoints 1000,10000      # Checkpoints custom
python run_experiments.py --resume results/<dir>        # Reprendre un run
python run_experiments.py --plot results/<dir>          # Re-générer les graphiques
```

## Contrôles dans la GUI

| Touche / Action | Rôle |
|----------------|------|
| **SPACE** | Pause / Reprendre |
| **N** | Avancer d'un pas (step-by-step) |
| **R** | Restart la partie |
| **F11** | Plein écran |
| **+/-** | Ajuster la vitesse (toujours disponible) |
| **←/→** ou **↑/↓** | Ajuster la vitesse (sauf LineWorld/GridWorld en mode humain) |
| **ESC** | Quitter |

### Contrôles par environnement (mode humain)

| Environnement | Touche | Action |
|---------------|--------|--------|
| LineWorld | **← / →** | Gauche / Droite |
| GridWorld | **↑ / ↓ / ← / →** | Haut / Bas / Gauche / Droite |
| TicTacToe | **1-9** | Case 1 à 9 |
| Quarto | **0-9, A-F** | Position ou pièce (hex) |

### Contrôles souris

| Environnement | Action | Rôle |
|---------------|--------|------|
| TicTacToe | **Clic souris** (case) | Jouer sur la case cliquée |
| Quarto | **Clic souris** (plateau) | Placer une pièce (phase PLACE) |
| Quarto | **Clic souris** (panel droit) | Donner une pièce (phase GIVE) |

### Boutons cliquables (Quarto)

| Bouton | Rôle |
|--------|------|
| **⏸ Pause** | Pause/reprendre |
| **▶\| Avancer** | Un pas |
| **↺ Restart** | Nouvelle partie |
