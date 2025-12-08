"""
NOTE: Currently the forces to the left and righta are completly fixed we can update it
so that our controller can add more force and relax the problem away from just a 3 input system
"""

from math import copysign
from pygame.math import Vector2
from pygame.locals import KEYDOWN,KEYUP,K_LEFT,K_RIGHT
from pygame.sprite import collide_rect
from pygame.event import Event
import random

import settings as config
import numpy as np
import casadi as cs



getsign = lambda x : copysign(1, x)

class dynamics:
    
    def __init__(self,
                 dt=config.DT,
                 gravity=config.GRAVITY,
                 max_vel=Vector2(config.PLAYER_MAX_SPEED, 100),
                 accel = .5,
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

        self.wind_std = .1
        self.max_wind = 0
        self.step_count = 0



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
            [0],
            [0],
            [ax_gain],
            [0]
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

    def wrap_position(self):
        """
        Wrap the horizontal position x across the screen.
        """
        x = self.x[0, 0]
        width = config.XWIN
        x = x % width

        self.x[0, 0] = x

    def randomize_wind(self):
        """
        Update wind as a bounded random walk:
        """
        delta = random.uniform(-self.wind_std, self.wind_std)
        self.wind += delta
        self.wind = max(-self.max_wind, min(self.max_wind, self.wind))


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

        self.wrap_position()


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
    def step(self, u_x):

        self.step_count += 1

        self.randomize_wind()
        u = np.array([[u_x]])
        d = np.array([[self.wind]])

        # Continuous dynamics: x_dot = A x + B u + E d + c
        x_dot = self.A @ self.x + self.B @ u + self.E @ d + self.c

        # Euler integration: x_next = x + dt * x_dot
        self.x = self.x + self.dt * x_dot

        # Velocity clipping
        self.clamp_velocity()

        px, py, vx, vy = self.get_state()
        return px, py, vx, vy, float(self.wind)

    def step_symbolic(self, X, U):
        """
        Discrete-time one-step dynamics for MPC using CasADi symbols.
        X: 4x1 CasADi vector (SX/MX)
        U: 2x1 CasADi vector (SX/MX)
        Returns X_next: 4x1 CasADi vector
        """
        # Convert constant matrices to CasADi DM
        A = cs.DM(self.A)  # 4x4
        B = cs.DM(self.B)  # 4x2
        E = cs.DM(self.E)  # 4x1
        c = cs.DM(self.c)  # 4x1

        d = self.wind  # scalar (float)

        # Continuous-time dynamics: x_dot = A X + B U + E d + c
        x_dot = cs.mtimes(A, X) + cs.mtimes(B, U) + E * d + c

        # Euler discretization: X_next = X + dt * x_dot
        X_next = X + self.dt * x_dot

        return X_next
