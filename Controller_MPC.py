from level import Level
from player import Player
import settings as config
import casadi as ca
import math
from dynamics import Dynamics

class PlatformState:
    x: float
    y: float
    width: float
    height: float
    breakable: bool
    has_bonus: bool
    bonus_force: float | None

class MPCController:
    def __init__(self, horizon = 10, dt = 1/60):
        self.horizon = horizon
        self.dt = dt
        self.model = Dynamics(dt)
        self.opti = ca.Opti()
        self.X = self.opti.variable(4, self.N+1)
        self.U = self.opti.variable(1, self.N)
        self.X0 = self.opti.parameter(4)
        self.X_target = self.opti.parameter(1)
        self.platforms = self.get_platforms()
        self.player_state = self.get_player_state()
        self.last_platform = None
        self.window_width, self.window_height = self.window.get_size()


    def _build_mpc(self):

        cost = 0

        for k in range(self.N):
            Xk = self.X[:, k]
            Uk = self.U[:, k]
            X_next = self.model.step(Xk, Uk)
            self.opti.subject_to(self.X[:, k+1] == X_next)

            cost += (X_next[0] - self.X_target)**2
            cost += 0.01 * Uk[0]**2
            self.opti.subject_to(Uk <= 1)
            self.opti.subject_to(Uk >= -1)

            self.opti.subject_to(self.X[2, k] <= self.model.max_vx)
            self.opti.subject_to(self.X[2, k] >= -self.model.max_vx)
            self.opti.subject_to(self.X[0, k] >= 0)  
            self.opti.subject_to(self.X[0, k] <= self.window_width)
            self.opti.subject_to(self.X[1, k] >= 0)
            self.opti.subject_to(self.X[1, k] <= self.window_height)
            self.opti.subject_to(self.X[1, k] <= 2 * self.window_height)
            for p in self.platforms:
                plat_top = p.y
                big_M = 10000
                self.opti.subject_to(
                    self.X[1, k] >= plat_top + big_M * (self.X[3, k] / vy_min)
                )

        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.minimize(cost)
        self.opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    def get_player_state(self):
        player = Player.instance
        p_state = [player.rect.x, player.rect.y, player._velocity.x, player._velocity.y] 

        return p_state
    
    def get_platforms(self): 
        level = Level.instance

        platforms = []
        for p in level.platforms:
            platforms.append(PlatformState(
                x=p.rect.x,
                y=p.rect.y,
                width=p.rect.width,
                height=p.rect.height,
                breakable=p.breakable,
                has_bonus=(p.bonus is not None),
                bonus_force=(p.bonus.force if p.bonus else None)
            ))

        return platforms

    def get_safe_platforms(self):
        safe = []
        for p in Level.instance.platforms:
            if not p.breakable:
                safe.append(PlatformState(
                    x=p.rect.x,
                    y=p.rect.y,
                    width=p.rect.width,
                    height=p.rect.height,
                    breakable=p.breakable,
                    has_bonus=(p.bonus is not None),
                    bonus_force=(p.bonus.force if p.bonus else None)
                ))
        return safe
 
    def choose_target_platform(self):
        px, py, vx, vy = self.get_player_state()
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


    def compute_control(self):
        X0_val = self.get_player_state()
        target = self.choose_target_platform()

        if not target:
            return 0

        target_center = target.x + target.width / 2

        self.opti.set_value(self.X0, X0_val)
        self.opti.set_value(self.X_target, target_center)

        try:
            sol = self.opti.solve()
            return float(sol.value(self.U[0, 0]))
        except:
            return 0


MPC = MPCController()

