"""
NOTE: Currently the forces to the left and righta are completly fixed we can update it
so that our controller can add more force and relax the problem away from just a 3 input system
"""

from math import copysign
from pygame.math import Vector2
from pygame.locals import KEYDOWN,KEYUP,K_LEFT,K_RIGHT
from pygame.sprite import collide_rect
from pygame.event import Event

import settings as config
import numpy as np


getsign = lambda x : copysign(1, x)

class dynamics:
    
    def __init__(self,
                 dt=1,
                 gravity=config.GRAVITY,
                 max_vel=Vector2(config.PLAYER_MAX_SPEED, 100),
                 accel = .8,
                 deccel =.6):

        self.dt = dt
        self.gravity = gravity
        self.max_vel = max_vel

        # State is 4x1 vector
        self.x = np.zeros((4, 1))

        self.x[0,0]=config.XWIN/2

        # Optional forces
        self.wind = 0.0     # horizontal wind force
        self.rocket = 0.0   # upward force


        #A matrix continuous time
        self.A = np.array(
            [[0,0,1,0],
            [0,0,0,1],
            [0,0,0,0],
            [0,0,0,0]
                ], dtype=float)


        ax_gain = accel / self.dt
        vy_gain = 0.0  

        self.B = np.array([
        [0,      0],
        [0,      0],
        [ax_gain, 0],
        [0,    vy_gain]
        ], dtype=float)

        self.E = np.array([
            [0],
            [0],
            [1],
            [0]
        ], dtype=float)

        self.c = np.array([
            [0],
            [0],
            [0],
            [gravity]
        ], dtype=float)


    def clamp_velocity(self):
        """Clamp vx, vy to the allowed maximum velocities."""
        # vx = x[2], vy = x[3]
        vx = self.x[2, 0]
        vy = self.x[3, 0]

        if self.max_vel is not None:
            max_vx = self.max_vel.x
            max_vy = self.max_vel.y

            vx = max(-max_vx, min(max_vx, vx))
            vy = max(-max_vy, min(max_vy, vy))

        self.x[2, 0] = vx
        self.x[3, 0] = vy

    def set_state(self, x, y, vx, vy):
        """Hard-set the internal state to match the sprite."""
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = vx
        self.x[3, 0] = vy

    def get_state(self):
        """Return (x, y, vx, vy) as simple Python floats."""
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0]),
        )

    def updateyv(self, force):
        self.x[3, 0] = force
    def step(self, u_x, u_y):
        u = np.array([[u_x], [u_y]])
        d = np.array([[self.wind]])

        # Continuous dynamics: x_dot = A x + B u + E d + c
        x_dot = self.A @ self.x + self.B @ u + self.E @ d + self.c

        # Euler integration: x_next = x + dt * x_dot
        self.x = self.x + self.dt * x_dot

        # Velocity clipping
        self.clamp_velocity()

        return self.get_state()

      