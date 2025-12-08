# -*- coding: utf-8 -*-
"""
    CopyLeft 2021 Michael Rouves

    This file is part of Pygame-DoodleJump.
    Pygame-DoodleJump is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Pygame-DoodleJump is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Pygame-DoodleJump. If not, see <https://www.gnu.org/licenses/>.
"""


import pygame, sys

from singleton import Singleton
from camera import Camera
from player import Player
from level import Level
import settings as config



class Game(Singleton):
    """
    A class to represent the game.

    used to manage game updates, draw calls and user input events.
    Can be access via Singleton: Game.instance .
    (Check Singleton design pattern for more info)
    """

    # constructor called on new instance: Game()
    def __init__(self) -> None:
        
        # ============= Initialisation =============
        self.__alive = True
        # Window / Render
        self.window = pygame.display.set_mode(config.DISPLAY,config.FLAGS)
        self.clock = pygame.time.Clock()
        self.wind = 0.0
        self.start_time = pygame.time.get_ticks()
        self.final_time = None
        # Instances
        self.camera = Camera()
        self.lvl = Level()
        self.player = Player(
            config.HALF_XWIN - config.PLAYER_SIZE[0]/2,# X POS
            config.HALF_YWIN + config.HALF_YWIN/2,#      Y POS
            *config.PLAYER_SIZE,# SIZE
            config.PLAYER_COLOR#  COLOR
        )
        self.start_player_y = self.player.rect.y
        # Finish line: 100 m above start (50 px per meter)
        self.px_per_meter = 50
        self.finish_height_m = 100          # CHANGED from 50 to 100
        self.finish_world_y = self.start_player_y - self.finish_height_m * self.px_per_meter

        # User Interface
        self.score = 0
        self.score_txt = config.SMALL_FONT.render("0 m",1,config.GRAY)
        self.score_pos = pygame.math.Vector2(10,10)

        self.gameover_txt = config.LARGE_FONT.render("Game Over",1,config.GRAY)
        self.gameover_rect = self.gameover_txt.get_rect(
            center=(config.HALF_XWIN,config.HALF_YWIN))
        # Win text
        self.game_won = False
        self.win_txt = config.LARGE_FONT.render("You Won", 1, config.GRAY)
        self.win_rect = self.win_txt.get_rect(
            center=(config.HALF_XWIN, config.HALF_YWIN))
		
    def close(self):
        self.__alive = False


    def reset(self):
        self.camera.reset()
        self.lvl.reset()
        self.player.reset()
        self.start_time = pygame.time.get_ticks() 
        self.start_player_y = self.player.rect.y 
        self.finish_world_y = self.start_player_y - self.finish_height_m * self.px_per_meter
        self.game_won = False
        self.final_time = None


    def _event_loop(self):
        # ---------- User Events ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                if event.key == pygame.K_RETURN and self.player.dead:
                    self.reset()
            # # self.player.handle_event(event)
            # self.player.handle_event_MPC_input(self.lvl)


    def _update_loop(self):
        # ----------- Update -----------\
        if self.player.dead:
            return
        wind = self.player.handle_event_MPC_input(self.lvl)
        self.wind = wind
        self.player.update()
        self.lvl.update()

        if not self.player.dead:
            self.camera.update(self.player.rect)

            # --- Distance from start (in "meters") --- (50 pixels per meter)
    
            dy_pixels = max(0, self.start_player_y - self.player.rect.y)
            self.score = dy_pixels // 50   

            self.score_txt = config.SMALL_FONT.render(
                f"{self.score} m", 1, config.GRAY
            )
            # Check win condition: reach 100 m
            if self.score >= 100 and not self.game_won:   # CHANGED from 50 to 100
                self.game_won = True
                # store final time at win
                elapsed_ms = pygame.time.get_ticks() - self.start_time
                self.final_time = elapsed_ms // 1000
                self.player.dead = True  # stops updates and triggers end screen
                return

    

    def _render_loop(self):
        # ----------- Display -----------
        self.window.fill(config.WHITE)
        self.lvl.draw(self.window)
        self.player.draw(self.window)

        # ----------- Checkered finish line at 100 m -----------
        # Convert world y → screen y using camera
        line_y = int(self.finish_world_y - self.camera.state.y)

        # Only draw if within (or near) screen
        if -50 <= line_y <= self.window.get_height() + 50:
            band_height = 30      # total banner height in pixels
            square_size = 10      # size of each check square
            top_y = line_y - band_height // 2

            window_width = self.window.get_width()

            # Draw checkered pattern
            for y in range(top_y, top_y + band_height, square_size):
                row = (y - top_y) // square_size
                for x in range(0, window_width, square_size):
                    col = x // square_size
                    # alternate colors like a checkerboard
                    if (row + col) % 2 == 0:
                        color = config.WHITE
                    else:
                        color = config.BLACK
                    rect = pygame.Rect(x, y, square_size, square_size)
                    pygame.draw.rect(self.window, color, rect)

            # Optional: add a thin center line for emphasis
            pygame.draw.line(
                self.window,
                config.BLACK,
                (0, line_y),
                (window_width, line_y),
                2
            )

            # Optional: label for clarity
            label = config.SMALL_FONT.render("100 m", True, config.BLACK)  # CHANGED text
            self.window.blit(label, (10, top_y - 20))
        # User Interface
        if self.player.dead:
            if self.game_won:
                self.window.blit(self.win_txt, self.win_rect)      # win screen
            else:
                self.window.blit(self.gameover_txt, self.gameover_rect)  # lose screen

        self.window.blit(self.score_txt, self.score_pos)  # score txt

        self._draw_wind_indicator()

        # Freeze timer once game is over (dead or won)
        if self.player.dead:
            if self.final_time is None:
                # if dead for some other reason and we didn't store yet
                elapsed_ms = pygame.time.get_ticks() - self.start_time
                self.final_time = elapsed_ms // 1000
            elapsed_sec = self.final_time
        else:

            elapsed_ms = pygame.time.get_ticks() - self.start_time
            elapsed_sec = elapsed_ms // 1000

        minutes = elapsed_sec // 60
        seconds = elapsed_sec % 60

        time_str = f"{minutes:02d}:{seconds:02d}" 

        time_txt = config.SMALL_FONT.render(time_str, True, config.GRAY)
        self.window.blit(time_txt, (10, 40))  
        pygame.display.update()
        
        self.clock.tick(config.FPS)

    def _draw_wind_indicator(self):
        """Draw a small box in the corner with an arrow showing wind direction."""
        box_w, box_h = 140, 60
        margin = 10

        # Top-right corner
        box_rect = pygame.Rect(
            self.window.get_width() - box_w - margin,
            margin,
            box_w,
            box_h
        )

        # Background + border
        pygame.draw.rect(self.window, config.LIGHT_GRAY, box_rect, border_radius=8)
        pygame.draw.rect(self.window, config.GRAY, box_rect, 2, border_radius=8)

        # Label
        label = config.SMALL_FONT.render("Wind", True, config.BLACK)
        self.window.blit(label, (box_rect.x + 8, box_rect.y + 5))

        # Arrow parameters
        cx = box_rect.centerx
        cy = box_rect.centery + 5  # a bit lower than the text

        max_arrow_len = box_w // 2 - 25   # leave some margin
        max_wind_display = 50.0           # should match your dynamics max_wind

        # Clamp wind to displayable range
        w = max(-max_wind_display, min(max_wind_display, float(self.wind)))
        arrow_len = 0 if max_wind_display == 0 else (w / max_wind_display) * max_arrow_len

        start_pos = (cx, cy)
        end_pos = (cx + arrow_len, cy)

        # Draw main arrow line
        pygame.draw.line(self.window, config.BLACK, start_pos, end_pos, 3)

        # Draw arrow head (only if non-zero wind)
        if abs(arrow_len) > 1e-2:
            # Direction of arrow
            dx = arrow_len
            dy = 0

            # Normalize direction
            length = abs(dx)
            ux = dx / length
            uy = dy / length  # 0

            # Size of arrow head
            head_len = 10
            head_width = 6

            # Tip of arrow
            tip_x = end_pos[0]
            tip_y = end_pos[1]

            # Perpendicular vector for arrowhead
            px = 0
            py = 1  # vertical

            left_x  = tip_x - ux * head_len + px * head_width
            left_y  = tip_y - uy * head_len + py * head_width
            right_x = tip_x - ux * head_len - px * head_width
            right_y = tip_y - uy * head_len - py * head_width

            pygame.draw.polygon(
                self.window,
                config.BLACK,
                [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]
            )

        # Optional: show numeric wind value under arrow
        wind_str = f"{self.wind:.1f}"
        val_txt = config.SMALL_FONT.render(wind_str, True, config.BLACK)
        val_rect = val_txt.get_rect(center=(cx, box_rect.bottom - 12))
        self.window.blit(val_txt, val_rect)

    def run(self):
        # ============= MAIN GAME LOOP =============
        while self.__alive:
            self._event_loop()
            self._update_loop()
            self._render_loop()
        pygame.quit()




if __name__ == "__main__":
    # ============= PROGRAM STARTS HERE =============
    game = Game()
    game.run()
