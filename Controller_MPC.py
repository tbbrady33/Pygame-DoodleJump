from level import Level
import settings as config
import casadi as ca
import math
from Dynamics import dynamics
from pygame.math import Vector2
import settings as config
import numpy as np

class PlatformState:
    def __init__(self,x,y,width,height,breakable):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.breakable = breakable

class MPCController:
    def __init__(self, level,horizon = 10, dt = config.DT, max_vel=Vector2(config.PLAYER_MAX_SPEED, 100)):
        self.horizon = horizon
        self.dt = dt
        self.levell = level
        self.model = dynamics(dt)
        self.opti = ca.Opti()
        self.X = self.opti.variable(4, self.horizon+1)
        self.U = self.opti.variable(1, self.horizon)
        self.X0 = self.opti.parameter(4)
        self.X_target = self.opti.parameter(1)
        self.platforms = self.get_platforms()
        self.last_platform = None
        self.window_width = config.XWIN
        self.window_height = config.YWIN
        self.max_vel = max_vel
   
        self._build_mpc()
    


    def _build_mpc(self):

        cost = 0

        for k in range(self.horizon):
            Xk = self.X[:, k]
            Uk = self.U[:, k]
            X_next = self.model.step_symbolic(Xk, Uk)
            self.opti.subject_to(self.X[:, k+1] == X_next)

            cost += (X_next[0] - self.X_target)**2
            cost += 0.01 * Uk[0]**2
            self.opti.subject_to(Uk <= 1)
            self.opti.subject_to(Uk >= -1)

            self.opti.subject_to(self.X[2, k] <= self.max_vel.x)
            self.opti.subject_to(self.X[2, k] >= -self.max_vel.x)
            self.opti.subject_to(self.X[0, k] >= 0)  
            self.opti.subject_to(self.X[0, k] <= self.window_width)
            self.opti.subject_to(self.X[1, k] >= 0)
            self.opti.subject_to(self.X[1, k] <= self.window_height)
            self.opti.subject_to(self.X[1, k] <= 2 * self.window_height)
            # for p in self.platforms:
            #     plat_top = p.y
            #     big_M = 10000
                # self.opti.subject_to(
                #     self.X[1, k] >= plat_top + big_M * (self.X[3, k] / vy_min)
                # )

        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.minimize(cost)
        self.opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    
    # def get_player_state(self):
    #     player = Player.instance
    #     p_state = [player.rect.x, player.rect.y, player._velocity.x, player._velocity.y] 

    #     return p_state
    
    def get_platforms(self): 

        platforms = []
        for p in self.levell.platforms:
            platforms.append(PlatformState(
                x=p.rect.x,
                y=p.rect.y,
                width=p.rect.width,
                height=p.rect.height,
                breakable=p.breakable,
            ))

        return platforms

    def get_safe_platforms(self):
        safe = []
        for p in self.levell.platforms:
            if not p.breakable:
                safe.append(PlatformState(
                    x=p.rect.x,
                    y=p.rect.y,
                    width=p.rect.width,
                    height=p.rect.height,
                    breakable=p.breakable,
                ))
        return safe
 
    def choose_target_platform(self, x, y):
        px, py = x, y
        safe = self.get_safe_platforms()
        best = None
        best_dist = float("inf")

        for p in safe:
            center = p.x + p.width / 2
            dx = center - px
            dy = p.y - py
            if dy > 0:
                d = math.sqrt(dx*dx + dy*dy)
                if d < best_dist:
                    best_dist = d
                    best = p

        return best


    def compute_control(self, state):

        X0_val = state
        target = self.choose_target_platform(state[0], state[1])

        if not target:
            return 0

        target_center = target.x + target.width / 2

        self.opti.set_value(self.X0, X0_val)
        self.opti.set_value(self.X_target, target_center)

        try:
            sol = self.opti.solve()
            print(sol)
            return float(sol.value(self.U[0, 0]))
        except:
            return 0


# MPC = MPCController()

