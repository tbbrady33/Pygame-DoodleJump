# -*- coding: utf-8 -*-
"""
    CopyLeft 2021 Michael Rouves
    ...
"""

from math import copysign
from pygame.math import Vector2
from pygame.locals import KEYDOWN, KEYUP, K_LEFT, K_RIGHT
from pygame.sprite import collide_rect
from pygame.event import Event
from Controller_MPC import MPCController
import numpy as np

from singleton import Singleton
from sprite import Sprite
from level import Level
import settings as config
from Dynamics import dynamics

# Return the sign of a number: getsign(-5) -> -1
getsign = lambda x: copysign(1, x)


class Player(Sprite, Singleton):
    """
    A class to represent the player.
    
    Manages player's input, physics (movement...).
    Can be accessed via Singleton: Player.instance.
    """

    def __init__(self, *args):
        # calling default Sprite constructor
        self.wind = 0

        Sprite.__init__(self, *args)
        self.__startrect = self.rect.copy()
        self.__maxvelocity = Vector2(config.PLAYER_MAX_SPEED, 100)
        self.__startspeed = 1.5

        self._velocity = Vector2()
        self._input = 0
        self._jumpforce = config.PLAYER_JUMPFORCE
        self._bonus_jumpforce = config.PLAYER_BONUS_JUMPFORCE

        self.gravity = config.GRAVITY
        self.accel = 0.8
        self.deccel = 0.9
        self.dead = False

        # Continuous dynamics model
        self.dynamics = dynamics(
            dt=config.DT,
            gravity=config.GRAVITY,
            accel=self.accel,
            deccel=self.deccel,
            max_vel=Vector2(config.PLAYER_MAX_SPEED, 100),
        )
        self.dynamics.set_state(self.rect.x, self.rect.y, 0.0, 0.0)

        # MPC controllers 
        self.mpc_fast = None
        self.mpc_smooth = None

        # Toggle between controllers:
        # False  -> climb_fast
        # True   -> energy_saving
        self.use_energy_saving = False  # flip this for different runs / videos

    def _fix_velocity(self) -> None:
        """Set player's velocity between max/min. Should be called in Player.update()."""
        self._velocity.y = min(self._velocity.y, self.__maxvelocity.y)
        self._velocity.y = round(max(self._velocity.y, -self.__maxvelocity.y), 2)
        self._velocity.x = min(self._velocity.x, self.__maxvelocity.x)
        self._velocity.x = round(max(self._velocity.x, -self.__maxvelocity.x), 2)

    def reset(self) -> None:
        "Called only when game restarts (after player death)."
        self._velocity = Vector2()
        self.rect = self.__startrect.copy()
        self.camera_rect = self.__startrect.copy()
        self.dead = False
        self.dynamics.set_state(self.rect.x, self.rect.y, 0.0, 0.0)

    def handle_event_MPC_input(self, level) -> float:
        """
        Called each frame from Game._update_loop() to compute the MPC control.
        `level` is passed from Game (self.lvl).
        """
        # Lazy-create controllers once we actually have the level object
        if self.mpc_fast is None or self.mpc_smooth is None:
            self.mpc_fast = MPCController(level, mode="climb_fast")
            self.mpc_smooth = MPCController(level, mode="energy_saving")

        # Pick which controller to use this run
        if self.use_energy_saving:
            mpc = self.mpc_smooth
        else:
            mpc = self.mpc_fast

        state = np.array(self.dynamics.get_state())
        self._input = mpc.compute_control(state)
        return self.wind

    """
    NOTE: This function only allows velocities to be -v0, v0 and 0,
    we might want to relax the problem to move it away from a MIQP
    """

    def handle_event(self, event: Event) -> None:
        """Manual keyboard input (currently not used if MPC is active)."""
        if event.type == KEYDOWN:
            # Moves player only on x-axis (left/right)
            if event.key == K_LEFT:
                self._velocity.x = -self.__startspeed
                self._input = -1
            elif event.key == K_RIGHT:
                self._velocity.x = self.__startspeed
                self._input = 1
        elif event.type == KEYUP:
            if (event.key == K_LEFT and self._input == -1) or (
                event.key == K_RIGHT and self._input == 1
            ):
                self._input = 0

    def jump(self, force=None) -> None:
        if force is None:
            force = self._jumpforce

        # Upward velocity
        self._velocity.y = -force

        # Also update dynamics
        self.dynamics.x[3, 0] = -force
        self.dynamics.x[0, 0] = self.rect.x
        self.dynamics.x[1, 0] = self.rect.y

    def onCollide(self, obj: Sprite) -> None:
        self.rect.bottom = obj.rect.top
        self.jump()
        self.dynamics.set_state(
            self.rect.x, self.rect.y, self._velocity.x, self._velocity.y
        )

        # Tell the point-collector MPC that we hit this platform
        if hasattr(self, "mpc_smooth"):
            try:
                self.mpc_smooth.mark_platform_visited(obj)
            except Exception:
                pass


    def collisions(self) -> None:
        """Checks for collisions with level. Should be called in Player.update()."""
        lvl = Level.instance
        if not lvl:
            return
        for platform in lvl.platforms:
            # check falling and colliding <=> isGrounded ?
            if self._velocity.y > 0.5:
                # check collisions with platform's spring bonus
                if platform.bonus and collide_rect(self, platform.bonus):
                    self.onCollide(platform.bonus)
                    self.jump(platform.bonus.force)
                    self.dynamics.set_state(
                        self.rect.x, self.rect.y, self._velocity.x, self._velocity.y
                    )

                # check collisions with platform
                if collide_rect(self, platform):
                    self.onCollide(platform)
                    platform.onCollide()

    def update(self):
        # die check
        if self.camera_rect.y > config.YWIN * 2:
            self.dead = True
            return

        # Use MPC-generated input
        u_x = self._input * self.accel
        u_y = 0  # no vertical thrust input

        px, py, vx, vy, wind = self.dynamics.step(u_x)
        self.wind = wind

        self.rect.x = px
        self.rect.y = py

        # Update velocity inside the Player object
        self._velocity = Vector2(vx, vy)

        # collision detection
        self.collisions()
