# DeepRL — Commandes disponibles

> Voir [SPECS.md](SPECS.md) pour la spécification technique complète (encoding, architecture, interfaces).

## Démos console (statistiques texte)

```bash
python main.py --env lineworld       # Random sur LineWorld + parties/sec
python main.py --env gridworld       # Random sur GridWorld + parties/sec
python main.py --env tictactoe       # Random vs Random sur TicTacToe + parties/sec
python main.py --env quarto          # Random vs Random sur Quarto + parties/sec
```

## Interface graphique — Observer un agent

```bash
python main.py --env lineworld --agent DDQN_ER        # Observer DDQN_ER sur LineWorld
python main.py --env gridworld --agent DDQN_PER       # Observer DDQN_PER sur GridWorld
python main.py --env tictactoe --agent DDQN_ER        # Observer DDQN_ER sur TicTacToe
python main.py --env quarto --agent DDQN_ER           # Observer DDQN_ER sur Quarto
python main.py --env tictactoe --agent Random         # Forcer l'agent Random
```

## Agent vs Agent (GUI)

```bash
python main.py --env tictactoe --agent DDQN_ER --versus DDQN_PER   # DDQN_ER vs DDQN_PER
python main.py --env quarto --agent AlphaZero --versus DDQN_ER     # AlphaZero vs DDQN_ER
```

## Jouer contre l'IA (Humain vs Agent)

```bash
python main.py --env lineworld --agent Human          # Jouer sur LineWorld (clavier)
python main.py --env gridworld --agent Human          # Jouer sur GridWorld (clavier)
python main.py --env tictactoe --agent Human          # Humain (J0) vs Random (J1)
python main.py --env tictactoe --agent Human --versus DDQN_ER   # Humain vs DDQN_ER
python main.py --env quarto --agent Human --versus AlphaZero    # Humain vs AlphaZero
```

## Humain vs Humain (même écran)

```bash
python main.py --env tictactoe --agent Human --versus Human    # 2 joueurs humains
python main.py --env quarto --agent Human --versus Human       # 2 joueurs humains
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
