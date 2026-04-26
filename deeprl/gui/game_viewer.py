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
    BLACK = (30, 30, 30)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (210, 215, 220)
    DARK_GRAY = (80, 80, 80)
    RED = (220, 60, 60)
    SOFT_RED = (240, 130, 130)
    GREEN = (60, 180, 60)
    SOFT_GREEN = (200, 235, 200)
    BLUE = (55, 90, 220)
    SOFT_BLUE = (190, 210, 245)
    YELLOW = (220, 220, 60)
    ORANGE = (255, 165, 0)
    BG_COLOR = (240, 242, 245)
    
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
        self._btn_pause_rect = None
        self._btn_step_rect = None
        self._btn_restart_rect = None
        self._restart_requested = False
        
        # Benchmark vitesse (random vs random)
        self.games_per_second = self._benchmark_speed()
        
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
        
        # Flag layout Quarto 3 colonnes
        self._quarto_3col = "quarto" in env_name
        
        if "line" in env_name:
            self.grid_width = self.env.size
            self.grid_height = 1
        elif "grid" in env_name:
            self.grid_width = self.env.width
            self.grid_height = self.env.height
        elif "tictactoe" in env_name:
            self.grid_width = 3
            self.grid_height = 3
            self.cell_size = max(self.cell_size, 130)
        elif self._quarto_3col:
            self.grid_width = 4
            self.grid_height = 4
            self.cell_size = max(self.cell_size, 140)
        else:
            self.grid_width = 5
            self.grid_height = 5
        
        if self._quarto_3col:
            # ── Layout 3 colonnes : [info gauche | plateau centre | pièces droite] ──
            self._q_left_width = 300        # panel info gauche
            self._q_board_margin = 30       # marge autour du plateau
            self._q_label_margin = 25       # espace pour les indices col/row
            board_px = 4 * self.cell_size
            self._q_board_area = board_px + 2 * self._q_board_margin + self._q_label_margin
            self._q_right_width = 320       # panel pièces droite
            self._q_ctrl_bar_h = 40         # barre de contrôles en bas
            
            self.game_width = self._q_left_width + self._q_board_area + self._q_right_width
            board_total = board_px + 2 * self._q_board_margin + self._q_label_margin + self._q_ctrl_bar_h
            self.game_height = max(board_total, 750)
            
            self.info_width = 0  # pas de panel info classique
            self.window_width = self.game_width
            self.window_height = self.game_height
        else:
            # Layout standard : jeu + panel info à droite
            self.game_width = self.grid_width * self.cell_size
            self.game_height = self.grid_height * self.cell_size
            self.info_width = 250
            self.window_width = self.game_width + self.info_width
            self.window_height = max(self.game_height, 520)
    
    def _benchmark_speed(self, n_games: int = 500) -> float:
        """Mesure la vitesse de simulation (parties/sec) avec un joueur random."""
        from deeprl.agents.random_agent import RandomAgent
        env = self.env.clone()
        rng_agent = RandomAgent(state_dim=env.state_dim, n_actions=env.n_actions)
        start = time.time()
        for _ in range(n_games):
            state = env.reset()
            while not env.is_game_over:
                available = env.get_available_actions()
                action = rng_agent.act(state, available)
                state, _, _ = env.step(action)
        elapsed = time.time() - start
        return n_games / elapsed if elapsed > 0 else 0.0
    
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
        self._restart_requested = False
        
        while self.running and not self.env.is_game_over:
            # Gérer les événements
            action = self._handle_events()
            
            if not self.running:
                break
            
            if self._restart_requested:
                self._restart_requested = False
                return  # Sort de l'épisode, la boucle run() en lance un nouveau
            
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
        
        # Fin d'épisode — pause automatique pour voir le résultat
        if self.env.is_game_over:
            self._update_stats()
            self.paused = True
            while self.paused and self.running:
                self._handle_events()
                if self._restart_requested:
                    self._restart_requested = False
                    self.paused = False
                    return
                self._render()
                self.clock.tick(30)
            self.paused = False
    
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
                elif event.key == pygame.K_r:
                    self._restart_requested = True
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                
                # Vitesse : +/- toujours, flèches seulement si
                # pas en mode humain sur un env qui utilise les flèches
                _arrows_for_env = (
                    self.agent is None
                    and any(k in self.env.name.lower() for k in ("line", "grid"))
                )
                if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS,
                                 pygame.K_EQUALS):
                    self.fps = min(60, self.fps + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.fps = max(1, self.fps - 1)
                elif not _arrows_for_env:
                    if event.key in (pygame.K_UP, pygame.K_RIGHT):
                        self.fps = min(60, self.fps + 1)
                    elif event.key in (pygame.K_DOWN, pygame.K_LEFT):
                        self.fps = max(1, self.fps - 1)
                
                # Contrôles pour les environnements
                if self.agent is None:  # Mode humain
                    action = self._key_to_action(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Boutons UI (Quarto 3-col)
                if self._handle_button_click(event.pos):
                    pass  # Action UI traitée
                elif self.agent is None:
                    action = self._mouse_to_action(event.pos)
        
        return action
    
    def _handle_button_click(self, pos: tuple) -> bool:
        """Gère les clics sur les boutons UI. Retourne True si un bouton a été cliqué."""
        if not getattr(self, '_quarto_3col', False):
            return False
        
        if self._btn_pause_rect and self._btn_pause_rect.collidepoint(pos):
            self.paused = not self.paused
            return True
        
        if self._btn_step_rect and self._btn_step_rect.collidepoint(pos):
            self.step_mode = True
            return True
        
        if self._btn_restart_rect and self._btn_restart_rect.collidepoint(pos):
            self._restart_requested = True
            return True
        
        return False
    
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
                raw = key_map[key]
                # Phase give : décaler de +16 pour obtenir l'action 16-31
                if self.env._phase == "give":
                    action = raw + 16
                else:
                    action = raw
                if action in self.env.get_available_actions():
                    return action
        
        return None
    
    def _mouse_to_action(self, pos: tuple) -> Optional[int]:
        """Convertit un clic souris en action."""
        x, y = pos
        env_name = self.env.name.lower()
        
        if self._quarto_3col and "quarto" in env_name:
            # ── Layout 3 colonnes ──
            cs = self.cell_size
            bx = self._q_left_width + self._q_board_margin + self._q_label_margin
            by = self._q_board_margin + self._q_label_margin
            board_px = 4 * cs
            
            if self.env._phase == "place":
                # Clic sur le plateau (zone centrale)
                if bx <= x < bx + board_px and by <= y < by + board_px:
                    c = (x - bx) // cs
                    r = (y - by) // cs
                    action = r * 4 + c
                    if action in self.env.get_available_actions():
                        return action
            else:
                # Phase give : clic sur une pièce dans le panel droit
                for piece_id, rect in self._quarto_piece_rects.items():
                    if rect.collidepoint(x, y):
                        action = piece_id + 16
                        if action in self.env.get_available_actions():
                            return action
            return None
        
        # Layouts standards
        if x >= self.game_width:
            return None
        
        col = x // self.cell_size
        row = y // self.cell_size
        
        if "tictactoe" in env_name:
            action = row * 3 + col
            if action in self.env.get_available_actions():
                return action
        
        return None
    
    def _render(self):
        """Affiche l'état actuel."""
        self.screen.fill(self.BG_COLOR)
        
        if self._quarto_3col:
            # Layout 3 colonnes Quarto
            self._draw_quarto_left_panel()
            self._draw_quarto()
            self._draw_quarto_right_panel()
            self._draw_quarto_ctrl_bar()
        else:
            # Layout standard
            self._draw_game()
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
        cs = self.cell_size
        y = self.window_height // 2 - cs // 2
        
        for i in range(self.env.size):
            x = i * cs
            rect = pygame.Rect(x, y, cs, cs)
            
            is_agent = (i == self.env._position)
            is_goal = (i == self.env.size - 1)
            is_fail = (i == 0)
            
            # Couleur de fond
            if is_agent:
                color = self.SOFT_BLUE
            elif is_goal:
                color = self.SOFT_GREEN
            elif is_fail:
                color = self.SOFT_RED
            else:
                color = self.WHITE
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.DARK_GRAY, rect, 2)
            
            # Symbole central
            cx, cy = x + cs // 2, y + cs // 2
            if is_agent:
                pygame.draw.circle(self.screen, self.BLUE, (cx, cy), cs // 3)
                txt = self.font.render("A", True, self.WHITE)
                self.screen.blit(txt, txt.get_rect(center=(cx, cy)))
            elif is_goal:
                txt = self.font.render("G", True, self.GREEN)
                self.screen.blit(txt, txt.get_rect(center=(cx, cy)))
            elif is_fail:
                txt = self.font.render("F", True, self.RED)
                self.screen.blit(txt, txt.get_rect(center=(cx, cy)))
            
            # Numéro de position
            num = self.font_small.render(str(i), True, self.GRAY)
            self.screen.blit(num, num.get_rect(center=(cx, y + cs + 18)))
    
    def _draw_gridworld(self):
        """Dessine GridWorld."""
        cs = self.cell_size
        # Déterminer la position fail (coin supérieur droit)
        fail_pos = (0, self.env.width - 1) if hasattr(self.env, 'width') else None
        
        for row in range(self.env.height):
            for col in range(self.env.width):
                x = col * cs
                y = row * cs
                rect = pygame.Rect(x, y, cs, cs)
                
                pos = (row, col)
                is_agent = (pos == self.env._agent_pos)
                is_goal = (pos == self.env.goal_pos)
                is_wall = (pos in self.env.walls)
                is_fail = (fail_pos is not None and pos == fail_pos)
                is_trap = hasattr(self.env, 'traps') and pos in self.env.traps
                
                # Couleur selon le type de case
                if is_agent:
                    color = self.SOFT_BLUE
                elif is_goal:
                    color = self.SOFT_GREEN
                elif is_fail or is_trap:
                    color = self.SOFT_RED
                elif is_wall:
                    color = self.GRAY
                else:
                    color = self.WHITE
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.DARK_GRAY, rect, 2)
                
                # Symboles
                cx, cy = x + cs // 2, y + cs // 2
                if is_agent:
                    pygame.draw.circle(self.screen, self.BLUE, (cx, cy), cs // 3)
                    text = self.font.render("A", True, self.WHITE)
                    self.screen.blit(text, text.get_rect(center=(cx, cy)))
                elif is_goal:
                    text = self.font.render("G", True, self.GREEN)
                    self.screen.blit(text, text.get_rect(center=(cx, cy)))
                elif is_fail or is_trap:
                    text = self.font.render("F", True, self.RED)
                    self.screen.blit(text, text.get_rect(center=(cx, cy)))
                elif is_wall:
                    text = self.font.render("#", True, self.WHITE)
                    self.screen.blit(text, text.get_rect(center=(cx, cy)))
    
    def _draw_tictactoe(self):
        """Dessine TicTacToe."""
        cs = self.cell_size
        board = self.env._board.reshape(3, 3)
        available = self.env.get_available_actions() if not self.env.is_game_over else []
        
        # Détecter la ligne gagnante
        winning_cells = self._get_tictactoe_winning_cells(board)
        
        for row in range(3):
            for col in range(3):
                x = col * cs
                y = row * cs
                rect = pygame.Rect(x + 2, y + 2, cs - 4, cs - 4)
                idx = row * 3 + col
                
                # Couleur de fond
                if (row, col) in winning_cells:
                    bg = (255, 245, 180)  # jaune clair pour la ligne gagnante
                elif idx in available and self.agent is None:
                    bg = self.SOFT_GREEN  # surbrillance cases jouables (mode humain)
                else:
                    bg = self.WHITE
                
                pygame.draw.rect(self.screen, bg, rect, border_radius=6)
                pygame.draw.rect(self.screen, self.DARK_GRAY, rect, 2, border_radius=6)
                
                # Symbole X / O
                val = board[row, col]
                cx, cy = x + cs // 2, y + cs // 2
                r = int(cs * 0.30)
                lw = max(5, cs // 18)  # épaisseur proportionnelle
                
                if val == 1:  # X
                    offset = r
                    pygame.draw.line(self.screen, self.BLUE,
                                     (cx - offset, cy - offset),
                                     (cx + offset, cy + offset), lw)
                    pygame.draw.line(self.screen, self.BLUE,
                                     (cx + offset, cy - offset),
                                     (cx - offset, cy + offset), lw)
                elif val == -1:  # O
                    pygame.draw.circle(self.screen, self.RED, (cx, cy), r, lw)
    
    @staticmethod
    def _get_tictactoe_winning_cells(board):
        """Retourne les positions de la ligne gagnante, ou un set vide."""
        lines = [
            [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
            [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
            [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
        ]
        for line in lines:
            vals = [board[r][c] for r, c in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return set(line)
        return set()
    
    def _draw_quarto_piece(self, cx, cy, piece_id, size):
        """
        Dessine une pièce Quarto en pseudo-3D centrée en (cx, cy).
        
        Attributs visuels:
        - tall/short → hauteur du corps
        - dark/light → couleur (brun foncé / beige)
        - solid/hollow → rempli / trou au sommet
        - square/round → carré / cylindre
        """
        tall = bool(piece_id & 1)
        dark = bool(piece_id & 2)
        solid = bool(piece_id & 4)
        square = bool(piece_id & 8)
        
        s = int(size)
        if s < 4:
            s = 4
        
        # Hauteur du corps : grand vs petit
        body_h = int(s * (1.8 if tall else 1.1))
        
        # Couleurs
        if dark:
            face_color = (120, 75, 35)
            side_color = (90, 55, 20)
            top_color = (139, 90, 43)
            outline = (70, 40, 12)
        else:
            face_color = (230, 205, 160)
            side_color = (200, 175, 130)
            top_color = (245, 222, 179)
            outline = (170, 140, 95)
        
        icx, icy = int(cx), int(cy)
        lw = max(1, int(s * 0.06))
        
        # Décalage perspective (pseudo 3D)
        persp = int(s * 0.25)
        
        if square:
            # ── Cube / boîte ──
            hw = s  # demi-largeur
            # Face avant (rectangle)
            front = pygame.Rect(icx - hw, icy - body_h // 2 + persp, hw * 2, body_h)
            pygame.draw.rect(self.screen, face_color, front)
            pygame.draw.rect(self.screen, outline, front, lw)
            
            # Face droite (parallélogramme)
            right_pts = [
                (icx + hw, icy - body_h // 2 + persp),
                (icx + hw + persp, icy - body_h // 2),
                (icx + hw + persp, icy - body_h // 2 + body_h - persp),
                (icx + hw, icy - body_h // 2 + persp + body_h),
            ]
            pygame.draw.polygon(self.screen, side_color, right_pts)
            pygame.draw.polygon(self.screen, outline, right_pts, lw)
            
            # Face du dessus (parallélogramme)
            top_pts = [
                (icx - hw, icy - body_h // 2 + persp),
                (icx - hw + persp, icy - body_h // 2),
                (icx + hw + persp, icy - body_h // 2),
                (icx + hw, icy - body_h // 2 + persp),
            ]
            pygame.draw.polygon(self.screen, top_color, top_pts)
            pygame.draw.polygon(self.screen, outline, top_pts, lw)
            
            # hollow : trou sur le dessus
            if not solid:
                hole_cx = icx + persp // 2
                hole_cy = icy - body_h // 2 + persp // 2
                hole_r = int(s * 0.35)
                dark_hole = (50, 30, 10) if dark else (150, 120, 80)
                pygame.draw.ellipse(
                    self.screen, dark_hole,
                    (hole_cx - hole_r, hole_cy - hole_r // 2, hole_r * 2, hole_r),
                )
                pygame.draw.ellipse(
                    self.screen, outline,
                    (hole_cx - hole_r, hole_cy - hole_r // 2, hole_r * 2, hole_r),
                    lw,
                )
        else:
            # ── Cylindre ──
            hw = s  # demi-largeur
            ell_h = int(s * 0.4)  # hauteur de l'ellipse
            
            top_cy = icy - body_h // 2
            bot_cy = icy + body_h // 2
            
            # Corps du cylindre (rectangle entre les deux ellipses)
            body_rect = pygame.Rect(icx - hw, top_cy, hw * 2, body_h)
            pygame.draw.rect(self.screen, face_color, body_rect)
            
            # Ellipse du bas
            pygame.draw.ellipse(
                self.screen, side_color,
                (icx - hw, bot_cy - ell_h // 2, hw * 2, ell_h),
            )
            pygame.draw.ellipse(
                self.screen, outline,
                (icx - hw, bot_cy - ell_h // 2, hw * 2, ell_h),
                lw,
            )
            
            # Bords verticaux
            pygame.draw.line(self.screen, outline, (icx - hw, top_cy), (icx - hw, bot_cy), lw)
            pygame.draw.line(self.screen, outline, (icx + hw, top_cy), (icx + hw, bot_cy), lw)
            
            # Ellipse du haut
            pygame.draw.ellipse(
                self.screen, top_color,
                (icx - hw, top_cy - ell_h // 2, hw * 2, ell_h),
            )
            pygame.draw.ellipse(
                self.screen, outline,
                (icx - hw, top_cy - ell_h // 2, hw * 2, ell_h),
                lw,
            )
            
            # hollow : trou au sommet
            if not solid:
                hole_r = int(s * 0.55)
                hole_ell_h = int(ell_h * 0.6)
                dark_hole = (50, 30, 10) if dark else (150, 120, 80)
                pygame.draw.ellipse(
                    self.screen, dark_hole,
                    (icx - hole_r, top_cy - hole_ell_h // 2, hole_r * 2, hole_ell_h),
                )
                pygame.draw.ellipse(
                    self.screen, outline,
                    (icx - hole_r, top_cy - hole_ell_h // 2, hole_r * 2, hole_ell_h),
                    lw,
                )
    
    def _draw_quarto(self):
        """
        Dessine le plateau Quarto 4×4 dans la zone centrale (layout 3 colonnes).
        
        Le plateau est entouré d'indices ligne/colonne.
        Phase PLACE : surbrillance verte sur les cases vides.
        """
        cs = self.cell_size
        board_px = 4 * cs
        
        # Origine du plateau dans la zone centrale
        bx = self._q_left_width + self._q_board_margin + self._q_label_margin
        by = self._q_board_margin + self._q_label_margin
        
        # Fond zone plateau (brun)
        board_bg = pygame.Rect(
            self._q_left_width, 0,
            self._q_board_area, self.window_height - self._q_ctrl_bar_h
        )
        pygame.draw.rect(self.screen, (140, 110, 75), board_bg)
        
        # Indices colonnes (au-dessus)
        for col in range(4):
            txt = self.font.render(str(col), True, (220, 210, 190))
            self.screen.blit(
                txt, txt.get_rect(center=(bx + col * cs + cs // 2, by - 18))
            )
        
        # Indices lignes (à gauche)
        for row in range(4):
            txt = self.font.render(str(row), True, (220, 210, 190))
            self.screen.blit(
                txt, txt.get_rect(center=(bx - 18, by + row * cs + cs // 2))
            )
        
        # ── Cases du plateau ──
        for row in range(4):
            for col in range(4):
                x = bx + col * cs
                y = by + row * cs
                pos = row * 4 + col
                
                piece_id = self.env._board[pos]
                
                # Fond de cellule
                margin = 4
                inner = pygame.Rect(x + margin, y + margin, cs - 2 * margin, cs - 2 * margin)
                
                if self.env._phase == "place" and piece_id < 0:
                    bg = (40, 140, 50)  # vert foncé = jouable
                else:
                    bg = (50, 130, 55) if piece_id < 0 else (45, 120, 50)
                
                pygame.draw.rect(self.screen, bg, inner, border_radius=8)
                
                if piece_id >= 0:
                    self._draw_quarto_piece(
                        x + cs // 2, y + cs // 2,
                        piece_id, cs * 0.28
                    )
                    # ID de la pièce en dessous
                    lbl = self.font_small.render(str(piece_id), True, (200, 230, 200))
                    self.screen.blit(lbl, lbl.get_rect(center=(x + cs // 2, y + cs - 14)))
    
    def _draw_quarto_left_panel(self):
        """Dessine le panel info à gauche pour le layout Quarto 3 colonnes.
        
        Utilise des positions Y FIXES pour éviter le scintillement
        quand _current_piece alterne entre None et une valeur.
        """
        pw = self._q_left_width
        x = 18
        
        # Fond du panel
        panel = pygame.Rect(0, 0, pw, self.window_height - self._q_ctrl_bar_h)
        pygame.draw.rect(self.screen, (65, 60, 55), panel)
        
        # ═══════ Zone 1 : Joueur + mode (y = 18..80) ═══════
        Y_PLAYER = 18
        
        # Déterminer qui est le joueur courant (humain ou IA)
        cp = self.env._current_player
        is_human_turn = False
        opponent_name = None
        if hasattr(self, 'opponent_agent'):
            # Mode HumanVsAgent
            opponent_name = self.opponent_agent.name
            is_human_turn = (cp == self.human_player)
        
        if is_human_turn:
            player_color = (100, 220, 100)
            player_label = f"Joueur {cp} — Votre tour"
        elif opponent_name and not is_human_turn:
            player_color = (220, 160, 80)
            player_label = f"Joueur {cp} — {opponent_name}"
        else:
            player_color = (100, 200, 100)
            player_label = f"Joueur {cp}"
        
        player_txt = self.font.render(player_label, True, player_color)
        # Tronquer si nécessaire
        if player_txt.get_width() > pw - 36:
            player_txt = self.font.render(f"J{cp} — {'Vous' if is_human_turn else opponent_name or ''}", True, player_color)
        self.screen.blit(player_txt, (x, Y_PLAYER))
        
        mode_name = self._get_quarto_mode_name(opponent_name)
        mode_txt = self.font_small.render(mode_name, True, (180, 180, 180))
        self.screen.blit(mode_txt, (x, Y_PLAYER + 32))
        
        # ═══════ Zone 2 : Phase + attributs (y = 82..140) ═══════
        Y_PHASE = 82
        pygame.draw.line(self.screen, (100, 95, 85), (x, Y_PHASE - 4), (pw - 18, Y_PHASE - 4), 1)
        
        if self.env._current_piece is not None:
            phase_str = f"J{self.env._current_player} — PLACE la pièce #{self.env._current_piece}"
        else:
            phase_str = f"J{self.env._current_player} — DONNE une pièce"
        
        desc_surf = self.font_small.render(phase_str, True, self.WHITE)
        if desc_surf.get_width() > pw - 36:
            # Tronquer
            short = "PLACE" if self.env._current_piece is not None else "DONNE"
            desc_surf = self.font_small.render(short, True, self.WHITE)
        self.screen.blit(desc_surf, (x, Y_PHASE))
        
        # Attributs (toujours affichés, ou "—" si pas de pièce)
        Y_ATTRS = Y_PHASE + 24
        if self.env._current_piece is not None:
            p = self.env._current_piece
            attrs = [
                "GRAND" if p & 1 else "COURT",
                "FONCÉ" if p & 2 else "CLAIR",
                "PLEIN" if p & 4 else "CREUX",
                "CARRÉ" if p & 8 else "ROND",
            ]
            attr_str = "-".join(attrs)
        else:
            attr_str = "—"
        attr_txt = self.font_small.render(attr_str, True, (180, 170, 150))
        self.screen.blit(attr_txt, (x, Y_ATTRS))
        
        # ═══════ Zone 3 : Aperçu pièce (y = 148..290) — TOUJOURS affichée ═══════
        Y_PREVIEW = 148
        pygame.draw.line(self.screen, (100, 95, 85), (x, Y_PREVIEW - 6), (pw - 18, Y_PREVIEW - 6), 1)
        
        label = self.font_small.render("Pièce à placer :", True, (150, 145, 135))
        self.screen.blit(label, (x, Y_PREVIEW))
        
        preview_rect = pygame.Rect(x, Y_PREVIEW + 26, pw - 36, 100)
        pygame.draw.rect(self.screen, (80, 75, 68), preview_rect, border_radius=8)
        
        if self.env._current_piece is not None:
            self._draw_quarto_piece(
                x + preview_rect.width // 2,
                Y_PREVIEW + 26 + 45,
                self.env._current_piece,
                28,
            )
            id_txt = self.font.render(
                str(self.env._current_piece), True, (200, 190, 170)
            )
            self.screen.blit(
                id_txt,
                id_txt.get_rect(center=(x + preview_rect.width // 2, Y_PREVIEW + 26 + 80)),
            )
        else:
            none_txt = self.font_small.render("Aucune", True, (120, 115, 108))
            self.screen.blit(
                none_txt,
                none_txt.get_rect(center=(x + preview_rect.width // 2, Y_PREVIEW + 26 + 50)),
            )
        
        # ═══════ Zone 4 : Boutons (y = 300..380) ═══════
        Y_BUTTONS = 300
        pygame.draw.line(self.screen, (100, 95, 85), (x, Y_BUTTONS - 6), (pw - 18, Y_BUTTONS - 6), 1)
        
        btn_w = 125
        btn_h = 34
        
        # ── Bouton Pause / Reprendre ──
        self._btn_pause_rect = pygame.Rect(x, Y_BUTTONS, btn_w, btn_h)
        if self.paused:
            pause_bg = (50, 130, 60)
            pause_label = "▶ Reprendre"
        else:
            pause_bg = (140, 90, 40)
            pause_label = "⏸ Pause"
        pygame.draw.rect(self.screen, pause_bg, self._btn_pause_rect, border_radius=5)
        pygame.draw.rect(self.screen, (180, 170, 155), self._btn_pause_rect, 1, border_radius=5)
        p_lbl = self.font_small.render(pause_label, True, self.WHITE)
        self.screen.blit(p_lbl, p_lbl.get_rect(center=self._btn_pause_rect.center))
        
        # ── Bouton Avancer (un pas) ──
        self._btn_step_rect = pygame.Rect(x + btn_w + 10, Y_BUTTONS, btn_w, btn_h)
        step_bg = (70, 120, 140) if self.paused else (80, 75, 68)
        pygame.draw.rect(self.screen, step_bg, self._btn_step_rect, border_radius=5)
        pygame.draw.rect(self.screen, (120, 150, 170) if self.paused else (100, 95, 85), self._btn_step_rect, 1, border_radius=5)
        s_lbl = self.font_small.render("▶| Avancer", True, self.WHITE if self.paused else (120, 115, 108))
        self.screen.blit(s_lbl, s_lbl.get_rect(center=self._btn_step_rect.center))
        
        # ── Bouton Restart ──
        Y_RESTART = Y_BUTTONS + btn_h + 10
        self._btn_restart_rect = pygame.Rect(x, Y_RESTART, btn_w, btn_h)
        pygame.draw.rect(self.screen, (100, 60, 55), self._btn_restart_rect, border_radius=5)
        pygame.draw.rect(self.screen, (150, 90, 80), self._btn_restart_rect, 1, border_radius=5)
        r_lbl = self.font_small.render("↺ Restart", True, self.WHITE)
        self.screen.blit(r_lbl, r_lbl.get_rect(center=self._btn_restart_rect.center))
        
        # ═══════ Zone 5 : Statistiques (y = 400..530) ═══════
        Y_STATS = 400
        pygame.draw.line(self.screen, (100, 95, 85), (x, Y_STATS - 6), (pw - 18, Y_STATS - 6), 1)
        
        stats_items = [
            (f"Tour : {self.step_count}", (180, 175, 165)),
            (f"Épisode : {self.episode_count + 1}", (180, 175, 165)),
            (f"Vitesse : {self.games_per_second:,.0f} p/s", (130, 170, 230)),
        ]
        for i, (txt, color) in enumerate(stats_items):
            surf = self.font_small.render(txt, True, color)
            self.screen.blit(surf, (x, Y_STATS + i * 24))
        
        # Scores
        Y_SCORES = Y_STATS + 85
        score_data = [
            ("V", str(self.wins), (100, 200, 100)),
            ("D", str(self.losses), (220, 100, 100)),
            ("N", str(self.draws), (180, 180, 180)),
        ]
        for i, (lbl, val, color) in enumerate(score_data):
            line = self.font_small.render(f"{lbl}: {val}", True, color)
            self.screen.blit(line, (x, Y_SCORES + i * 22))
        
        # ═══════ Zone 6 : État du jeu (bas du panel) ═══════
        Y_STATUS = self.window_height - self._q_ctrl_bar_h - 75
        if self.paused and not self.env.is_game_over:
            p_txt = self.font.render("⏸ PAUSE", True, self.ORANGE)
            self.screen.blit(p_txt, (x, Y_STATUS))
        elif self.env.is_game_over:
            if hasattr(self.env, '_winner'):
                if self.env._winner is not None and self.env._winner >= 0:
                    win_color = (100, 220, 100)
                    e_txt = self.font.render(
                        f"Joueur {self.env._winner} gagne !", True, win_color
                    )
                else:
                    e_txt = self.font.render("MATCH NUL", True, (180, 180, 180))
            else:
                e_txt = self.font.render("FIN", True, self.GRAY)
            self.screen.blit(e_txt, (x, Y_STATUS))
            # Indication pour continuer
            hint = self.font_small.render(
                "[R] Restart  [SPACE] Continuer", True, (150, 145, 135)
            )
            self.screen.blit(hint, (x, Y_STATUS + 32))
    
    def _draw_quarto_right_panel(self):
        """Dessine le panel de pièces disponibles à droite."""
        pw = self._q_right_width
        rx = self._q_left_width + self._q_board_area
        panel = pygame.Rect(rx, 0, pw, self.window_height - self._q_ctrl_bar_h)
        pygame.draw.rect(self.screen, (65, 60, 55), panel)
        
        x = rx + 18
        y = 18
        
        # Titre
        title = self.font.render("Pièces disponibles", True, self.WHITE)
        self.screen.blit(title, (x, y))
        y += 32
        
        if self.env._phase == "give":
            hint = self.font_small.render("↓ Clique pour choisir", True, (180, 170, 150))
            self.screen.blit(hint, (x, y))
        y += 28
        
        # Grille 4×4 de pièces
        piece_size = 22
        col_spacing = (pw - 36) // 4
        row_h = int((self.window_height - self._q_ctrl_bar_h - y - 20) / 4)
        row_h = min(row_h, 140)
        
        self._quarto_piece_rects = {}
        
        for i in range(16):
            col = i % 4
            row_off = i // 4
            cx = x + col * col_spacing + col_spacing // 2
            cy = y + row_off * row_h + row_h // 2
            
            available = i in self.env._available_pieces
            
            cell_rect = pygame.Rect(
                cx - col_spacing // 2 + 4,
                cy - row_h // 2 + 4,
                col_spacing - 8,
                row_h - 8,
            )
            
            if available:
                self._quarto_piece_rects[i] = cell_rect
                if self.env._phase == "give":
                    pygame.draw.rect(self.screen, (40, 100, 45), cell_rect, border_radius=8)
                    pygame.draw.rect(self.screen, (60, 150, 65), cell_rect, 2, border_radius=8)
                else:
                    pygame.draw.rect(self.screen, (75, 70, 63), cell_rect, border_radius=8)
                    pygame.draw.rect(self.screen, (100, 95, 85), cell_rect, 1, border_radius=8)
                self._draw_quarto_piece(cx, cy - 5, i, piece_size)
            else:
                pygame.draw.rect(self.screen, (55, 52, 48), cell_rect, border_radius=8)
                # Indicateur pièce utilisée
                id_used = self.font_small.render(f"#{i}", True, (90, 85, 78))
                self.screen.blit(id_used, id_used.get_rect(center=(cx, cy)))
            
            # Étiquette ID
            lbl_color = self.WHITE if available else (90, 85, 78)
            lbl = self.font_small.render(str(i), True, lbl_color)
            self.screen.blit(lbl, lbl.get_rect(center=(cx, cy + row_h // 2 - 16)))
    
    def _draw_quarto_ctrl_bar(self):
        """Dessine la barre de contrôles en bas de la fenêtre Quarto."""
        bar_y = self.window_height - self._q_ctrl_bar_h
        bar_rect = pygame.Rect(0, bar_y, self.window_width, self._q_ctrl_bar_h)
        pygame.draw.rect(self.screen, (50, 47, 42), bar_rect)
        pygame.draw.line(
            self.screen, (80, 75, 68), (0, bar_y), (self.window_width, bar_y), 1
        )
        
        ctrls = (
            "[R] Restart  [N] Avancer  [+/-] Vitesse: "
            f"{1.0/self.fps:.1f}s  [F11] Plein écran  "
            "[SPACE] Pause  [ESC] Quitter"
        )
        txt = self.font_small.render(ctrls, True, (170, 165, 155))
        self.screen.blit(txt, (15, bar_y + 10))
    
    def _draw_info_panel(self):
        """Dessine le panel d'informations."""
        if getattr(self, '_quarto_3col', False):
            return
        px = self.game_width
        # Fond du panel
        panel_rect = pygame.Rect(px, 0, self.info_width, self.window_height)
        pygame.draw.rect(self.screen, self.WHITE, panel_rect)
        pygame.draw.line(self.screen, self.LIGHT_GRAY, (px, 0), (px, self.window_height), 2)
        
        x = px + 15
        y = 15
        
        # Titre
        title = self.font.render("DeepRL", True, self.BLUE)
        self.screen.blit(title, (x, y))
        y += 45
        
        # Environnement
        env_text = self.font_small.render(f"Env: {self.env.name}", True, self.BLACK)
        self.screen.blit(env_text, (x, y))
        y += 28
        
        # Agent
        y = self._draw_agent_info(x, y)
        
        # Vitesse de simulation
        speed_text = self.font_small.render(
            f"Vitesse: {self.games_per_second:,.0f} parties/s", True, self.BLUE
        )
        self.screen.blit(speed_text, (x, y))
        y += 35
        
        # Séparateur
        pygame.draw.line(self.screen, self.LIGHT_GRAY, (x, y), (x + self.info_width - 30, y), 2)
        y += 18
        
        stats = [
            ("Épisode:", str(self.episode_count + 1)),
            ("Step:", str(self.step_count)),
            ("Reward:", f"{self.total_reward:.2f}"),
        ]
        for label, val in stats:
            lbl = self.font_small.render(label, True, self.GRAY)
            v = self.font_small.render(val, True, self.BLACK)
            self.screen.blit(lbl, (x, y))
            self.screen.blit(v, (x + 80, y))
            y += 24
        
        y += 10
        score_stats = [
            ("Victoires:", str(self.wins), self.GREEN),
            ("Défaites:", str(self.losses), self.RED),
            ("Nuls:", str(self.draws), self.GRAY),
        ]
        for label, val, color in score_stats:
            lbl = self.font_small.render(label, True, self.DARK_GRAY)
            v = self.font_small.render(val, True, color)
            self.screen.blit(lbl, (x, y))
            self.screen.blit(v, (x + 90, y))
            y += 24
        
        # Séparateur
        y += 10
        pygame.draw.line(self.screen, self.LIGHT_GRAY, (x, y), (x + self.info_width - 30, y), 2)
        y += 15
        
        controls = [
            "SPACE: Pause",
            "N: Step-by-step",
            "+/-: Vitesse",
            "F11: Plein écran",
            "ESC: Quitter",
            f"FPS: {self.fps}",
        ]
        
        header = self.font_small.render("Contrôles", True, self.DARK_GRAY)
        self.screen.blit(header, (x, y))
        y += 22
        for ctrl in controls:
            ctrl_text = self.font_small.render(ctrl, True, self.GRAY)
            self.screen.blit(ctrl_text, (x + 8, y))
            y += 20
        
        # Contrôles spécifiques Quarto
        if "quarto" in self.env.name.lower():
            y += 8
            quarto_ctrls = [
                "Quarto:",
                "Clic: placer/donner",
                "0-9, A-F: selection",
            ]
            for qc in quarto_ctrls:
                self.screen.blit(self.font_small.render(qc, True, self.GRAY), (x + 8, y))
                y += 20
        
        # État du jeu (en bas du panel avec fond coloré)
        state_y = self.window_height - 50
        state_rect = pygame.Rect(px + 1, state_y - 5, self.info_width - 2, 45)

        if self.paused:
            pygame.draw.rect(self.screen, (255, 240, 200), state_rect)
            pause_text = self.font.render("PAUSE", True, self.ORANGE)
            self.screen.blit(pause_text, (x, state_y))
        elif self.env.is_game_over:
            if hasattr(self.env, '_winner'):
                if self.env._winner is not None and self.env._winner >= 0:
                    pygame.draw.rect(self.screen, (200, 240, 200), state_rect)
                    end_text = self.font.render(f"Joueur {self.env._winner} gagne!", True, self.GREEN)
                else:
                    pygame.draw.rect(self.screen, self.LIGHT_GRAY, state_rect)
                    end_text = self.font.render("MATCH NUL", True, self.GRAY)
            else:
                pygame.draw.rect(self.screen, self.LIGHT_GRAY, state_rect)
                end_text = self.font.render("FIN", True, self.GRAY)
            self.screen.blit(end_text, (x, state_y))
    
    def _draw_agent_info(self, x: int, y: int) -> int:
        """Dessine les infos agent dans le panel. Surchargeable par les sous-classes."""
        if self.agent:
            agent_text = self.font_small.render(f"Agent: {self.agent.name}", True, self.BLACK)
        else:
            agent_text = self.font_small.render("Mode: Humain", True, self.GREEN)
        self.screen.blit(agent_text, (x, y))
        return y + 28

    def _get_quarto_mode_name(self, opponent_name: str) -> str:
        """Retourne le label de mode pour le panel Quarto. Surchargeable."""
        if self.agent:
            return f"Observation: {self.agent.name}"
        elif opponent_name:
            return f"Humain vs {opponent_name}"
        return "Humain"

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


class AgentVsAgentViewer(GameViewer):
    """
    Viewer pour observer deux agents s'affronter.

    J0: agent_0, J1: agent_1. Les deux agents sont des IA.
    Utile pour comparer deux agents entraînés face à face.
    """

    def __init__(
        self,
        env: Environment,
        agent_0: Agent,
        agent_1: Agent,
        cell_size: int = 80,
        fps: int = 2,
        title: str = "Agent vs Agent"
    ):
        super().__init__(
            env=env,
            agent=agent_0,
            cell_size=cell_size,
            fps=fps,
            title=title
        )
        self.agent_0 = agent_0
        self.agent_1 = agent_1

    # ── Hooks d'affichage ────────────────────────────────────────────────────

    def _draw_agent_info(self, x: int, y: int) -> int:
        t0 = self.font_small.render(f"J0: {self.agent_0.name}", True, self.BLACK)
        t1 = self.font_small.render(f"J1: {self.agent_1.name}", True, self.GRAY)
        self.screen.blit(t0, (x, y))
        self.screen.blit(t1, (x, y + 22))
        return y + 50

    def _get_quarto_mode_name(self, opponent_name: str) -> str:
        return f"{self.agent_0.name} vs {self.agent_1.name}"

    # ── Logique de jeu ───────────────────────────────────────────────────────

    def _run_episode(self):
        """Execute un episode en alternant entre agent_0 (J0) et agent_1 (J1)."""
        state = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self._restart_requested = False

        while self.running and not self.env.is_game_over:
            self._handle_events()

            if not self.running:
                break
            if self._restart_requested:
                self._restart_requested = False
                return
            if self.paused and not self.step_mode:
                self._render()
                self.clock.tick(30)
                continue

            # Dispatcher vers le bon agent selon le joueur courant
            current = self.env.current_player
            agent = self.agent_0 if current == 0 else self.agent_1
            available = self.env.get_available_actions()

            if hasattr(agent, 'n_simulations'):
                action = agent.act(state, available, training=False, env=self.env)
            else:
                action = agent.act(state, available, training=False)

            self.step_mode = False
            state, reward, done = self.env.step(action)
            self.total_reward += reward
            self.step_count += 1

            self._render()
            self.clock.tick(self.fps)

        if self.env.is_game_over:
            self._update_stats()
            self.paused = True
            while self.paused and self.running:
                self._handle_events()
                if self._restart_requested:
                    self._restart_requested = False
                    self.paused = False
                    return
                self._render()
                self.clock.tick(30)
            self.paused = False


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
        self._restart_requested = False
        
        while self.running and not self.env.is_game_over:
            if self._restart_requested:
                self._restart_requested = False
                return
            
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
            
            if not self.running or self._restart_requested:
                break
            
            if action is not None:
                # Executer l'action
                state, reward, done = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1
            
            # Afficher
            self._render()
            self.clock.tick(self.fps)
        
        # Fin d'episode — pause automatique pour voir le résultat
        if self.env.is_game_over:
            self._update_stats_versus()
            self.paused = True
            while self.paused and self.running:
                self._handle_events()
                if self._restart_requested:
                    self._restart_requested = False
                    self.paused = False
                    return
                self._render()
                self.clock.tick(30)
            self.paused = False
    
    def _wait_for_human_action(self) -> Optional[int]:
        """Attend une action valide de l'humain."""
        while self.running:
            action = self._handle_events()
            
            if not self.running or self._restart_requested:
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
        if getattr(self, '_quarto_3col', False):
            return
        super()._draw_info_panel()
        
        x = self.game_width + 15
        y = self.window_height - 95
        
        # Afficher qui joue
        if not self.env.is_game_over:
            current = self.env.current_player
            if current == self.human_player:
                turn_text = self.font.render("Votre tour", True, self.GREEN)
            else:
                turn_text = self.font.render("Tour IA...", True, self.ORANGE)
            self.screen.blit(turn_text, (x, y))


def watch_agent_vs_agent(
    env: Environment,
    agent_0: Agent,
    agent_1: Agent,
    n_episodes: int = 10,
    fps: int = 2
):
    """
    Observe deux agents s'affronter (J0: agent_0, J1: agent_1).

    Args:
        env: Environnement 2 joueurs (TicTacToe, Quarto)
        agent_0: Agent jouant en premier (joueur 0)
        agent_1: Agent jouant en second (joueur 1)
        n_episodes: Nombre de parties
        fps: Vitesse d'affichage
    """
    if not PYGAME_AVAILABLE:
        print("[WARN] pygame non disponible. Installez-le avec: pip install pygame")
        return

    viewer = AgentVsAgentViewer(
        env=env,
        agent_0=agent_0,
        agent_1=agent_1,
        fps=fps,
        title=f"{agent_0.name} vs {agent_1.name}"
    )

    print(f"[GAME] {agent_0.name} (J0) vs {agent_1.name} (J1)")
    viewer.run(n_episodes=n_episodes)


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
