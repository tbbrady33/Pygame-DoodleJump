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

        self.wind_std = .2
        self.max_wind = 5
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
        return

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

        
    def step_symbolic(self, X, U, plat_y_vec, plat_left_vec, plat_right_vec):
        A = cs.DM(self.A)
        B = cs.DM(self.B)   # 4x1
        E = cs.DM(self.E)
        c = cs.DM(self.c)

        d = self.wind

        X_dot  = A @ X + B @ U + E * d + c
        X_cont = X + self.dt * X_dot

        x      = X[0]
        y      = X[1]
        vx     = X[2]
        vy     = X[3]

        x_next  = X_cont[0]
        y_next  = X_cont[1]
        vx_next = X_cont[2]
        vy_next = X_cont[3]

        N = plat_y_vec.numel()

        y_after  = y_next
        vy_after = vy_next

        # Pygame: vy > 0 = going down
        moving_down = vy > 0

        jump_vy = -float(config.PLAYER_JUMPFORCE)
        player_h = float(config.PLAYER_SIZE[1])

        for i in range(N):
            plat_top_y  = plat_y_vec[i]
            plat_x_left = plat_left_vec[i]
            plat_x_right= plat_right_vec[i]

            in_x = cs.logic_and(x >= plat_x_left, x <= plat_x_right)

            # treat y as TOP of sprite, add height to approximate bottom crossing platform top
            was_above = (y + player_h) <= plat_top_y
            now_below = (y_next + player_h) >= plat_top_y
            crossed_y = cs.logic_and(was_above, now_below)

            contact_i = cs.logic_and(moving_down, cs.logic_and(in_x, crossed_y))

            y_after  = cs.if_else(contact_i, plat_top_y, y_after)
            vy_after = cs.if_else(contact_i, jump_vy,             vy_after)

        X_next = cs.vertcat(x_next, y_after, vx_next, vy_after)
        return X_next
