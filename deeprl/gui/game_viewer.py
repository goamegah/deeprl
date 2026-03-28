"""
Game Viewer - Interface graphique pour visualiser les jeux.

Utilise Pygame pour afficher:
- L'état du jeu en temps réel
- Les actions de l'agent
- Les statistiques de performance

Permet aussi de jouer en tant qu'humain contre les agents.
"""

import numpy as np
from typing import Optional
import time

# Import conditionnel de pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("/!\\ pygame non installé. Installez-le avec: pip install pygame")

from deeprl.envs.base import Environment
from deeprl.agents.base import Agent


class GameViewer:
    """
    Visualiseur de jeux avec Pygame.
    
    Supporte:
    - Affichage graphique des environnements
    - Jeu humain (clavier/souris)
    - Observation des agents
    - Pause, vitesse variable
    
    Exemple d'utilisation:
        >>> viewer = GameViewer(env, agent)
        >>> viewer.run()  # Lance la fenêtre
    """
    
    # Couleurs
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    RED = (220, 60, 60)
    GREEN = (60, 180, 60)
    BLUE = (60, 60, 220)
    YELLOW = (220, 220, 60)
    ORANGE = (255, 165, 0)
    
    def __init__(
        self,
        env: Environment,
        agent: Optional[Agent] = None,
        cell_size: int = 80,
        fps: int = 5,
        title: str = "DeepRL Viewer"
    ):
        """
        Crée un visualiseur de jeu.
        
        Args:
            env: Environnement à afficher
            agent: Agent à observer (None pour mode humain)
            cell_size: Taille d'une cellule en pixels
            fps: Images par seconde (vitesse de jeu)
            title: Titre de la fenêtre
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame n'est pas installé")
        
        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        self.fps = fps
        self.title = title
        
        # État du viewer
        self.running = False
        self.paused = False
        self.step_mode = False  # Avancer step par step
        
        # Statistiques
        self.episode_count = 0
        self.total_reward = 0
        self.step_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        # Quarto-specific layout data
        self._quarto_piece_rects = {}
        self._quarto_panel_height = 0
        
        # Calculer la taille de la fenêtre
        self._calculate_window_size()
        
        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
    
    def _calculate_window_size(self):
        """Calcule la taille de la fenêtre selon l'environnement."""
        env_name = self.env.name.lower()
        
        if "line" in env_name:
            # LineWorld: une ligne horizontale
            self.grid_width = self.env.size
            self.grid_height = 1
        elif "grid" in env_name:
            # GridWorld: grille 2D
            self.grid_width = self.env.width
            self.grid_height = self.env.height
        elif "tictactoe" in env_name:
            # TicTacToe: 3x3
            self.grid_width = 3
            self.grid_height = 3
        elif "quarto" in env_name:
            # Quarto: 4x4 board + pieces panel
            self.grid_width = 4
            self.grid_height = 4
            self.cell_size = max(self.cell_size, 110)  # min 110px pour lisibilité
        else:
            # Défaut
            self.grid_width = 5
            self.grid_height = 5
        
        # Dimensions en pixels
        self.game_width = self.grid_width * self.cell_size
        self.game_height = self.grid_height * self.cell_size
        
        # Espace supplémentaire pour le panel de pièces Quarto
        if "quarto" in self.env.name.lower():
            self._quarto_panel_height = 380
            self.game_height += self._quarto_panel_height
        
        # Panel d'info à droite
        self.info_width = 280 if "quarto" in self.env.name.lower() else 250
        
        self.window_width = self.game_width + self.info_width
        self.window_height = max(self.game_height, 400)
    
    def init_pygame(self):
        """Initialise Pygame."""
        pygame.init()
        pygame.display.set_caption(self.title)
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
    
    def run(self, n_episodes: Optional[int] = None):
        """
        Lance le visualiseur.
        
        Args:
            n_episodes: Nombre d'épisodes à jouer (None = infini)
        """
        self.init_pygame()
        self.running = True
        
        episode = 0
        while self.running:
            if n_episodes is not None and episode >= n_episodes:
                break
            
            self._run_episode()
            episode += 1
            self.episode_count += 1
        
        pygame.quit()
    
    def _run_episode(self):
        """Exécute un épisode complet."""
        state = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        
        while self.running and not self.env.is_game_over:
            # Gérer les événements
            action = self._handle_events()
            
            if not self.running:
                break
            
            if self.paused and not self.step_mode:
                self._render()
                self.clock.tick(30)
                continue
            
            # Obtenir l'action
            if action is None:  # Pas d'action humaine
                if self.agent is not None:
                    available = self.env.get_available_actions()
                    
                    # Passer l'environnement pour MCTS si nécessaire
                    if hasattr(self.agent, 'n_simulations'):  # C'est MCTS
                        action = self.agent.act(state, available, training=False, env=self.env)
                    else:
                        action = self.agent.act(state, available, training=False)
                else:
                    # Mode humain, attendre l'input
                    self._render()
                    self.clock.tick(30)
                    continue
            
            if action is not None:
                # Exécuter l'action
                state, reward, done = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1
                
                self.step_mode = False
            
            # Afficher
            self._render()
            self.clock.tick(self.fps)
        
        # Fin d'épisode
        if self.env.is_game_over:
            self._update_stats()
            self._render()
            time.sleep(1.5)  # Pause pour voir le résultat
    
    def _handle_events(self) -> Optional[int]:
        """
        Gère les événements Pygame.
        
        Returns:
            Action choisie par l'humain, ou None
        """
        action = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n:
                    self.step_mode = True
                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 1)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 1)
                
                # Contrôles pour les environnements
                if self.agent is None:  # Mode humain
                    action = self._key_to_action(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.agent is None:
                    action = self._mouse_to_action(event.pos)
        
        return action
    
    def _key_to_action(self, key: int) -> Optional[int]:
        """Convertit une touche en action."""
        env_name = self.env.name.lower()
        
        if "line" in env_name:
            # LineWorld: gauche/droite
            if key == pygame.K_LEFT:
                return 0
            elif key == pygame.K_RIGHT:
                return 1
        
        elif "grid" in env_name:
            # GridWorld: 4 directions
            if key == pygame.K_UP:
                return 0
            elif key == pygame.K_DOWN:
                return 1
            elif key == pygame.K_LEFT:
                return 2
            elif key == pygame.K_RIGHT:
                return 3
        
        elif "tictactoe" in env_name:
            # TicTacToe: pavé numérique ou 1-9
            key_map = {
                pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
                pygame.K_4: 3, pygame.K_5: 4, pygame.K_6: 5,
                pygame.K_7: 6, pygame.K_8: 7, pygame.K_9: 8,
                pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2,
                pygame.K_KP4: 3, pygame.K_KP5: 4, pygame.K_KP6: 5,
                pygame.K_KP7: 6, pygame.K_KP8: 7, pygame.K_KP9: 8,
            }
            if key in key_map:
                action = key_map[key]
                if action in self.env.get_available_actions():
                    return action
        
        elif "quarto" in env_name:
            # Quarto: 0-9 et a-f pour sélectionner position/pièce (hex 0-15)
            key_map = {
                pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
                pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7,
                pygame.K_8: 8, pygame.K_9: 9,
                pygame.K_a: 10, pygame.K_b: 11, pygame.K_c: 12,
                pygame.K_d: 13, pygame.K_e: 14, pygame.K_f: 15,
            }
            if key in key_map:
                action = key_map[key]
                if action in self.env.get_available_actions():
                    return action
        
        return None
    
    def _mouse_to_action(self, pos: tuple) -> Optional[int]:
        """Convertit un clic souris en action."""
        x, y = pos
        
        # Vérifier qu'on est dans la zone de jeu
        if x >= self.game_width:
            return None
        
        col = x // self.cell_size
        row = y // self.cell_size
        
        env_name = self.env.name.lower()
        
        if "tictactoe" in env_name:
            action = row * 3 + col
            if action in self.env.get_available_actions():
                return action
        
        elif "quarto" in env_name:
            board_px = 4 * self.cell_size
            if self.env._phase == "place":
                # Clic sur une case vide du plateau
                if x < board_px and y < board_px:
                    c = x // self.cell_size
                    r = y // self.cell_size
                    action = r * 4 + c
                    if action in self.env.get_available_actions():
                        return action
            else:
                # Phase give : clic sur une pièce dans le panel
                for piece_id, rect in self._quarto_piece_rects.items():
                    if rect.collidepoint(x, y):
                        if piece_id in self.env.get_available_actions():
                            return piece_id
        
        return None
    
    def _render(self):
        """Affiche l'état actuel."""
        self.screen.fill(self.WHITE)
        
        # Dessiner le jeu
        self._draw_game()
        
        # Dessiner le panel d'info
        self._draw_info_panel()
        
        pygame.display.flip()
    
    def _draw_game(self):
        """Dessine l'état du jeu."""
        env_name = self.env.name.lower()
        
        if "line" in env_name:
            self._draw_lineworld()
        elif "grid" in env_name:
            self._draw_gridworld()
        elif "tictactoe" in env_name:
            self._draw_tictactoe()
        elif "quarto" in env_name:
            self._draw_quarto()
    
    def _draw_lineworld(self):
        """Dessine LineWorld."""
        y = self.window_height // 2 - self.cell_size // 2
        
        for i in range(self.env.size):
            x = i * self.cell_size
            rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            
            # Couleur de fond
            if i == self.env._position:
                color = self.BLUE
            elif i == self.env.size - 1:
                color = self.GREEN
            else:
                color = self.LIGHT_GRAY
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2)
            
            # Label
            text = self.font_small.render(str(i), True, self.BLACK)
            text_rect = text.get_rect(center=(x + self.cell_size//2, y + self.cell_size + 20))
            self.screen.blit(text, text_rect)
    
    def _draw_gridworld(self):
        """Dessine GridWorld."""
        for row in range(self.env.height):
            for col in range(self.env.width):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                pos = (row, col)
                
                # Couleur selon le type de case
                if pos == self.env._agent_pos:
                    color = self.BLUE
                elif pos == self.env.goal_pos:
                    color = self.GREEN
                elif pos in self.env.walls:
                    color = self.GRAY
                elif hasattr(self.env, 'traps') and pos in self.env.traps:
                    color = self.RED
                else:
                    color = self.LIGHT_GRAY
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 2)
                
                # Symboles
                center = (x + self.cell_size//2, y + self.cell_size//2)
                if pos == self.env._agent_pos:
                    text = self.font.render("A", True, self.WHITE)
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
                elif pos == self.env.goal_pos:
                    text = self.font.render("G", True, self.WHITE)
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
    
    def _draw_tictactoe(self):
        """Dessine TicTacToe."""
        board = self.env._board.reshape(3, 3)
        
        for row in range(3):
            for col in range(3):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Fond
                pygame.draw.rect(self.screen, self.LIGHT_GRAY, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 3)
                
                # Symbole
                val = board[row, col]
                center = (x + self.cell_size//2, y + self.cell_size//2)
                radius = self.cell_size // 3
                
                if val == 1:  # X
                    offset = radius - 5
                    pygame.draw.line(self.screen, self.BLUE, 
                                   (center[0]-offset, center[1]-offset),
                                   (center[0]+offset, center[1]+offset), 4)
                    pygame.draw.line(self.screen, self.BLUE,
                                   (center[0]+offset, center[1]-offset),
                                   (center[0]-offset, center[1]+offset), 4)
                elif val == -1:  # O
                    pygame.draw.circle(self.screen, self.RED, center, radius, 4)
    
    def _draw_quarto_piece(self, cx, cy, piece_id, size):
        """
        Dessine une pièce Quarto centrée en (cx, cy).
        
        Attributs visuels:
        - tall/short → taille du symbole
        - dark/light → couleur (brun foncé / beige)
        - solid/hollow → rempli / contour seul
        - square/round → carré / cercle
        """
        tall = bool(piece_id & 1)
        dark = bool(piece_id & 2)
        solid = bool(piece_id & 4)
        square = bool(piece_id & 8)
        
        # Facteur de taille : grand vs petit
        s = int(size * (1.0 if tall else 0.6))
        if s < 3:
            s = 3
        
        # Couleurs
        if dark:
            fill_color = (139, 90, 43)       # brun foncé
            outline_color = (90, 55, 20)
        else:
            fill_color = (245, 222, 179)      # beige
            outline_color = (180, 150, 100)
        
        lw = max(2, int(size * 0.12))  # épaisseur du contour
        icx, icy = int(cx), int(cy)
        
        if square:
            rect = pygame.Rect(icx - s, icy - s, s * 2, s * 2)
            if solid:
                pygame.draw.rect(self.screen, fill_color, rect)
            pygame.draw.rect(self.screen, outline_color, rect, lw)
        else:
            if solid:
                pygame.draw.circle(self.screen, fill_color, (icx, icy), s)
            pygame.draw.circle(self.screen, outline_color, (icx, icy), s, lw)
    
    def _draw_quarto(self):
        """
        Dessine le plateau Quarto 4×4 et le panel de pièces disponibles.
        
        Phase PLACE : le joueur clique sur une case vide du plateau.
        Phase GIVE  : le joueur clique sur une pièce disponible dans le panel.
        """
        cs = self.cell_size
        board_px = 4 * cs
        
        # ── Plateau 4×4 ──
        for row in range(4):
            for col in range(4):
                x = col * cs
                y = row * cs
                pos = row * 4 + col
                rect = pygame.Rect(x, y, cs, cs)
                
                piece_id = self.env._board[pos]
                
                # Fond : surbrillance verte pour les cases jouables
                if self.env._phase == "place" and piece_id < 0:
                    bg = (235, 245, 235)
                else:
                    bg = self.LIGHT_GRAY
                
                pygame.draw.rect(self.screen, bg, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 2)
                
                if piece_id >= 0:
                    self._draw_quarto_piece(
                        x + cs // 2, y + cs // 2,
                        piece_id, cs * 0.38
                    )
                else:
                    # Numéro de position
                    txt = self.font_small.render(str(pos), True, self.GRAY)
                    self.screen.blit(txt, txt.get_rect(center=(x + cs // 2, y + cs // 2)))
        
        # ── Panel sous le plateau ──
        py = board_px + 8
        
        # Phase et joueur
        phase_str = "PLACER" if self.env._phase == "place" else "DONNER"
        player_str = f"Joueur {self.env._current_player + 1}"
        info_line = f"{player_str} — {phase_str}"
        self.screen.blit(self.font_small.render(info_line, True, self.BLACK), (5, py))
        py += 24
        
        # Pièce courante à placer
        if self.env._current_piece is not None:
            self.screen.blit(
                self.font_small.render("Piece:", True, self.BLACK), (8, py)
            )
            self._draw_quarto_piece(80, py + 12, self.env._current_piece, 18)
            # Attributs texte
            p = self.env._current_piece
            attrs = ""
            attrs += "G" if p & 1 else "p"
            attrs += "F" if p & 2 else "c"
            attrs += "P" if p & 4 else "x"
            attrs += "C" if p & 8 else "r"
            self.screen.blit(
                self.font_small.render(f"[{attrs}]  id={p}", True, self.GRAY), (110, py)
            )
            py += 32
        else:
            py += 4
        
        # Titre du panel de pièces
        if self.env._phase == "give":
            self.screen.blit(
                self.font_small.render("Choisir une piece a donner:", True, self.GREEN),
                (5, py),
            )
        else:
            self.screen.blit(
                self.font_small.render("Pieces disponibles:", True, self.GRAY),
                (5, py),
            )
        py += 24
        
        # Grille de pièces (4 lignes × 4 colonnes)
        piece_spacing = max(1, (board_px - 16) // 4)
        piece_draw_size = cs * 0.20
        row_height = 65
        
        self._quarto_piece_rects = {}
        
        for i in range(16):
            col = i % 4
            row_off = i // 4
            cx = 8 + col * piece_spacing + piece_spacing // 2
            cy = py + row_off * row_height + 26
            
            available = i in self.env._available_pieces
            
            cell_rect = pygame.Rect(
                cx - piece_spacing // 2 + 4,
                cy - 22,
                piece_spacing - 8,
                52,
            )
            
            if available:
                self._quarto_piece_rects[i] = cell_rect
                if self.env._phase == "give":
                    pygame.draw.rect(self.screen, (225, 245, 225), cell_rect)
                    pygame.draw.rect(self.screen, self.GREEN, cell_rect, 2)
                else:
                    pygame.draw.rect(self.screen, self.WHITE, cell_rect)
                    pygame.draw.rect(self.screen, self.GRAY, cell_rect, 1)
                self._draw_quarto_piece(cx, cy, i, piece_draw_size)
            else:
                pygame.draw.rect(self.screen, (240, 240, 240), cell_rect)
                pygame.draw.line(
                    self.screen, self.GRAY,
                    (cell_rect.left + 2, cell_rect.top + 2),
                    (cell_rect.right - 2, cell_rect.bottom - 2), 1,
                )
            
            # Étiquette hexadécimale
            lbl_color = self.BLACK if available else self.GRAY
            lbl = self.font_small.render(f"{i:X}", True, lbl_color)
            self.screen.blit(lbl, lbl.get_rect(center=(cx, cy + 28)))
    
    def _draw_info_panel(self):
        """Dessine le panel d'informations."""
        x = self.game_width + 10
        y = 10
        
        # Titre
        title = self.font.render("DeepRL", True, self.BLACK)
        self.screen.blit(title, (x, y))
        y += 40
        
        # Environnement
        env_text = self.font_small.render(f"Env: {self.env.name}", True, self.BLACK)
        self.screen.blit(env_text, (x, y))
        y += 25
        
        # Agent
        if self.agent:
            agent_text = self.font_small.render(f"Agent: {self.agent.name}", True, self.BLACK)
        else:
            agent_text = self.font_small.render("Mode: Humain", True, self.BLUE)
        self.screen.blit(agent_text, (x, y))
        y += 35
        
        # Statistiques
        pygame.draw.line(self.screen, self.GRAY, (x, y), (x + self.info_width - 20, y))
        y += 15
        
        stats = [
            f"Épisode: {self.episode_count + 1}",
            f"Step: {self.step_count}",
            f"Reward: {self.total_reward:.2f}",
            f"",
            f"Victoires: {self.wins}",
            f"Défaites: {self.losses}",
            f"Nuls: {self.draws}",
        ]
        
        for stat in stats:
            if stat:
                stat_text = self.font_small.render(stat, True, self.BLACK)
                self.screen.blit(stat_text, (x, y))
            y += 22
        
        # Contrôles
        y += 20
        pygame.draw.line(self.screen, self.GRAY, (x, y), (x + self.info_width - 20, y))
        y += 15
        
        controls = [
            "Contrôles:",
            "SPACE: Pause",
            "N: Step-by-step",
            "↑/↓: Vitesse",
            "ESC: Quitter",
            f"FPS: {self.fps}",
        ]
        
        for ctrl in controls:
            ctrl_text = self.font_small.render(ctrl, True, self.GRAY)
            self.screen.blit(ctrl_text, (x, y))
            y += 22
        
        # Contrôles spécifiques Quarto
        if "quarto" in self.env.name.lower():
            y += 5
            quarto_ctrls = [
                "Quarto:",
                "Clic: placer/donner",
                "0-9, A-F: selection",
            ]
            for qc in quarto_ctrls:
                self.screen.blit(self.font_small.render(qc, True, self.GRAY), (x, y))
                y += 22
        
        # État
        y += 20
        if self.paused:
            pause_text = self.font.render("PAUSE", True, self.ORANGE)
            self.screen.blit(pause_text, (x, y))
        elif self.env.is_game_over:
            if hasattr(self.env, '_winner'):
                if self.env._winner == 0:
                    end_text = self.font.render("VICTOIRE!", True, self.GREEN)
                elif self.env._winner == 1:
                    end_text = self.font.render("DÉFAITE", True, self.RED)
                else:
                    end_text = self.font.render("NUL", True, self.GRAY)
            else:
                end_text = self.font.render("FIN", True, self.GRAY)
            self.screen.blit(end_text, (x, y))
    
    def _update_stats(self):
        """Met à jour les statistiques en fin d'épisode."""
        if hasattr(self.env, '_winner'):
            if self.env._winner == 0:
                self.wins += 1
            elif self.env._winner == 1:
                self.losses += 1
            else:
                self.draws += 1
        elif self.total_reward > 0:
            self.wins += 1
        elif self.total_reward < 0:
            self.losses += 1
        else:
            self.draws += 1


class HumanVsAgentViewer(GameViewer):
    """
    Viewer special pour le mode Humain vs Agent.
    
    Permet a un humain de jouer contre un agent IA dans les jeux
    a 2 joueurs (TicTacToe, Quarto).
    
    L'humain joue avec la souris/clavier, l'agent repond automatiquement.
    """
    
    def __init__(
        self,
        env: Environment,
        opponent_agent: Agent,
        human_first: bool = True,
        cell_size: int = 80,
        fps: int = 30,
        title: str = "Humain vs Agent"
    ):
        """
        Cree un viewer Humain vs Agent.
        
        Args:
            env: Environnement de jeu (TicTacToe, Quarto, etc.)
            opponent_agent: Agent adversaire
            human_first: Si True, l'humain joue en premier
            cell_size: Taille d'une cellule en pixels
            fps: Images par seconde
            title: Titre de la fenetre
        """
        super().__init__(
            env=env,
            agent=None,  # On gere l'agent manuellement
            cell_size=cell_size,
            fps=fps,
            title=title
        )
        
        self.opponent_agent = opponent_agent
        self.human_first = human_first
        self.human_player = 0 if human_first else 1
        self.agent_player = 1 if human_first else 0
    
    def _run_episode(self):
        """Execute un episode avec alternance humain/agent."""
        state = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        
        while self.running and not self.env.is_game_over:
            # Determiner qui joue
            current_player = self.env.current_player
            
            if current_player == self.human_player:
                # Tour de l'humain
                action = self._wait_for_human_action()
            else:
                # Tour de l'agent
                action = self._get_agent_action(state)
                # Petite pause pour que l'humain voie l'action
                time.sleep(0.5)
            
            if not self.running:
                break
            
            if action is not None:
                # Executer l'action
                state, reward, done = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1
            
            # Afficher
            self._render()
            self.clock.tick(self.fps)
        
        # Fin d'episode
        if self.env.is_game_over:
            self._update_stats_versus()
            self._render()
            time.sleep(2.0)  # Pause plus longue pour voir le resultat
    
    def _wait_for_human_action(self) -> Optional[int]:
        """Attend une action valide de l'humain."""
        while self.running:
            action = self._handle_events()
            
            if not self.running:
                return None
            
            if self.paused:
                self._render()
                self.clock.tick(30)
                continue
            
            if action is not None:
                # Verifier que l'action est valide
                if action in self.env.get_available_actions():
                    return action
            
            self._render()
            self.clock.tick(self.fps)
        
        return None
    
    def _get_agent_action(self, state: np.ndarray) -> int:
        """Obtient l'action de l'agent."""
        available = self.env.get_available_actions()
        
        # Passer l'environnement pour MCTS si necessaire
        if hasattr(self.opponent_agent, 'n_simulations'):
            return self.opponent_agent.act(
                state, available, training=False, env=self.env
            )
        else:
            return self.opponent_agent.act(state, available, training=False)
    
    def _update_stats_versus(self):
        """Met a jour les stats en mode versus."""
        if hasattr(self.env, '_winner'):
            if self.env._winner == self.human_player:
                self.wins += 1
            elif self.env._winner == self.agent_player:
                self.losses += 1
            else:
                self.draws += 1
    
    def _draw_info_panel(self):
        """Dessine le panel d'info avec indication du tour."""
        # Appeler le parent d'abord
        super()._draw_info_panel()
        
        x = self.game_width + 10
        y = self.window_height - 100
        
        # Afficher qui joue
        if not self.env.is_game_over:
            current = self.env.current_player
            if current == self.human_player:
                turn_text = self.font.render("Votre tour", True, self.GREEN)
            else:
                turn_text = self.font.render("Tour IA...", True, self.ORANGE)
            self.screen.blit(turn_text, (x, y))


def play_human_vs_agent(
    env: Environment,
    agent: Agent,
    n_games: int = 5,
    human_first: bool = True
):
    """
    Lance une session de jeu humain contre agent.
    
    Pour les jeux a 2 joueurs (TicTacToe, Quarto), l'humain joue
    contre l'agent de maniere alternee.
    
    Args:
        env: Environnement de jeu (doit etre un jeu 2 joueurs)
        agent: Agent adversaire
        n_games: Nombre de parties
        human_first: Si True, l'humain joue en premier
    """
    if not PYGAME_AVAILABLE:
        print("[WARN] pygame non disponible. Installez-le avec: pip install pygame")
        return
    
    # Creer un viewer special pour le mode versus
    viewer = HumanVsAgentViewer(
        env=env,
        opponent_agent=agent,
        human_first=human_first,
        fps=30,
        title=f"Humain vs {agent.name}"
    )
    
    print(f"[GAME] Jouer contre {agent.name}")
    print("   Utilisez les touches ou la souris pour jouer.")
    if human_first:
        print("   Vous jouez en premier (X).")
    else:
        print("   L'agent joue en premier, vous etes O.")
    
    viewer.run(n_episodes=n_games)


def watch_agent(
    env: Environment,
    agent: Agent,
    n_episodes: int = 10,
    fps: int = 3
):
    """
    Observe un agent jouer.
    
    Args:
        env: Environnement
        agent: Agent à observer
        n_episodes: Nombre d'épisodes
        fps: Vitesse d'affichage
    """
    if not PYGAME_AVAILABLE:
        print("/!\\ pygame non disponible.")
        return
    
    viewer = GameViewer(
        env=env,
        agent=agent,
        fps=fps,
        title=f"Observation: {agent.name}"
    )
    
    print(f"O:O Observation de {agent.name}")
    viewer.run(n_episodes=n_episodes)


# Test
if __name__ == "__main__":
    print("=== Test du Game Viewer ===\n")
    
    if not PYGAME_AVAILABLE:
        print("pygame non installé, test ignoré.")
    else:
        from deeprl.envs import GridWorld, TicTacToe
        from deeprl.agents import RandomAgent
        
        print("1. Test avec GridWorld:")
        env = GridWorld.create_simple(5)
        agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
        
        viewer = GameViewer(env, agent, fps=3)
        print(f"   Fenêtre: {viewer.window_width}x{viewer.window_height}")
        
        print("\n2. Test avec TicTacToe:")
        env = TicTacToe()
        viewer = GameViewer(env, agent=None, fps=10)
        print(f"   Fenêtre: {viewer.window_width}x{viewer.window_height}")
        
        print("\nPour lancer l'interface:")
        print("   python -m deeprl.gui.game_viewer")
        
        print("\n[OK] Tests passés!")
