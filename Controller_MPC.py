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
        self.max_platforms = config.MAX_PLAT
        self.plat_top_y   = self.opti.parameter(self.max_platforms)
        self.plat_x_left  = self.opti.parameter(self.max_platforms)
        self.plat_x_right = self.opti.parameter(self.max_platforms)
        self.stop_weight = self.opti.parameter()
        self.ref_center_x = self.opti.parameter()
        self.ref_top_y    = self.opti.parameter()
        self.ref_weight   = self.opti.parameter()

   
        self.platforms = self.get_platforms()
        self.last_platform = None
        self.window_width = config.XWIN
        self.window_height = config.YWIN
        self.max_vel = max_vel
        
        self._build_mpc()
    


    def _build_mpc(self):
        """
        Build MPC problem with:
        - platform-aware dynamics
        - smooth 'potential field' attraction to vertically reachable platforms
        - penalty on being low (large y)
        - small penalty on control effort
        - extra term to be stationary above the closest platform below (when enabled)
        """

        cost = 0.0

        # Tuning parameters
        u_max  = 5.0
        qy     = 1e-2
        r      = 1e-4
        w_plat = 1e-1

        # Length scales (in screen pixels) 
        R_x = 300.0
        R_y = 250.0

        # ---- Vertical reach parameters in SCREEN coords ----
        y0  = self.X0[1]
        vy0 = self.X0[3]

        g_abs = abs(config.GRAVITY) + 1e-6

        # Dynamic jump reach from vertical speed (vy0 < 0 means going up)
        max_jump_up_dyn = (vy0**2) / (2.0 * g_abs)

        # Fall reach – how far below we still care about platforms
        max_fall_down  = 800.0  

        for k in range(self.horizon):
            # State and input at step k
            Xk = self.X[:, k]
            Uk = self.U[:, k]

            # One-step prediction with platform-aware dynamics
            X_next = self.model.step_symbolic(
                Xk,
                Uk,
                self.plat_top_y,
                self.plat_x_left,
                self.plat_x_right
            )
            self.opti.subject_to(self.X[:, k+1] == X_next)

            x_next  = X_next[0]
            y_next  = X_next[1]
            vx_next = X_next[2]

            ##Base cost terms

            # 1) Penalty on being low (Pygame: larger y = lower on screen)
            cost += qy * (y_next**2)

            # 2) Small control effort penalty
            cost += r * (Uk[0]**2)

            # 3) be stationary above the closest platform BELOW
            dx_ref     = x_next - self.ref_center_x
            dy_ref     = y_next - self.ref_top_y
            dist2_ref  = dx_ref**2 + dy_ref**2

            # Penalize being far from that platform center AND having horizontal speed
            cost += self.ref_weight * (dist2_ref + vx_next**2)

            # --- Platform potential field with phase-dependent reachability ---

            for i in range(self.max_platforms):
                plat_y   = self.plat_top_y[i]
                plat_l   = self.plat_x_left[i]
                plat_r   = self.plat_x_right[i]

                center_x = 0.5 * (plat_l + plat_r)

                dx_plat = x_next - center_x
                dy_plat = y_next - plat_y

                # Smooth squared distance in (x, y) with length scales
                dist2   = (dx_plat / R_x)**2 + (dy_plat / R_y)**2
                attract = ca.exp(-dist2)    # biggest when we're near the platform

                # ---- Vertical reachability mask (based on initial y0, vy0) ----
                dy_up0   = y0 - plat_y
                dy_down0 = plat_y - y0

                # Case 1: going UP (jump phase) – use max_jump_up_dyn
                cond_jump_reach = ca.logic_and(dy_up0 >= 0, dy_up0 <= max_jump_up_dyn)

                # Case 2: FALLING -> platforms BELOW within fall window
                cond_fall_reach = ca.logic_and(dy_down0 >= 0, dy_down0 <= max_fall_down)

                # vy0 < 0 : going up, vy0 >= 0 : falling
                cond_up_phase   = (vy0 < 0)
                cond_down_phase = (vy0 >= 0)

                cond_vert = ca.logic_or(
                    ca.logic_and(cond_up_phase,   cond_jump_reach),
                    ca.logic_and(cond_down_phase, cond_fall_reach)
                )

                # Only vertically reachable platforms create an attractive well
                well = ca.if_else(cond_vert,
                                -w_plat * attract,  # reachable → reward being near
                                0.0)                # unreachable → no effect

                cost += well

            # --- Constraints ---

            # Input bounds
            self.opti.subject_to(Uk[0] <= u_max)
            self.opti.subject_to(Uk[0] >= -u_max)

            # Velocity bounds
            self.opti.subject_to(self.X[2, k] <= self.max_vel.x)
            self.opti.subject_to(self.X[2, k] >= -self.max_vel.x)

            # Horizontal position bounds
            self.opti.subject_to(self.X[0, k] >= 0)
            self.opti.subject_to(self.X[0, k] <= self.window_width)

        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.X0)

        # Set objective
        self.opti.minimize(cost)

        # Solver
        self.opti.solver("ipopt", {"print_time": False}, {"print_level": 1})



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

    


    def compute_control(self, state):
        # 1) Set initial state parameter - pygame cords
        X0_val = np.array(state).flatten()
        self.opti.set_value(self.X0, X0_val)

        # Unpack current player state - pygame cords
        px, py, vx, vy = map(float, state)

        g_abs = abs(config.GRAVITY) + 1e-6

        # Dynamic jump reach from vertical speed 
        max_jump_up_dyn = (vy**2) / (2.0 * g_abs)

        # Fall reach
        max_fall_down = 800.0

        # highest-below selection when going up
        max_below_for_fallback = 1000.0  

        # Phase decision: going up vs falling 
        going_up = (vy < 0.0)

        above_cands = []  
        below_cands = []  

        for p in self.levell.platforms:
            plat_y = float(p.rect.y)
            plat_l = float(p.rect.x)
            plat_r = float(p.rect.x + p.rect.width)
            center_x = 0.5 * (plat_l + plat_r)

            dy_up   = py - plat_y   
            dy_down = plat_y - py    
            dx      = center_x - px

            if going_up:
                # ----- UP PHASE -----

               
                if dy_up > 0 and dy_up <= max_jump_up_dyn:
                    # We want the HIGHEST reachable platforms above:
                    score_y  = plat_y
                    score_dx = abs(dx)
                    above_cands.append((score_y, score_dx, plat_y, plat_l, plat_r))
                    continue

                # 2) Below candidate for fallback (when no above reachable)
                if dy_down > 0 and dy_down <= max_below_for_fallback:
                    # choose highest below by y, then |dx|
                    score_y  = plat_y
                    score_dx = abs(dx)
                    below_cands.append((score_y, score_dx, plat_y, plat_l, plat_r))
                    continue

            else:
                # ----- DOWN PHASE -----
                if dy_down > 0 and dy_down <= max_fall_down:
                    score_y  = plat_y
                    score_dx = abs(dx)
                    below_cands.append((score_y, score_dx, plat_y, plat_l, plat_r))

        # Is there any reachable-above platform this jump?
        has_above_reachable = len(above_cands) > 0
        if going_up and (not has_above_reachable) and below_cands:
            # sort by height (smallest plat_y = highest), then |dx|
            below_sorted = sorted(below_cands, key=lambda t: (t[0], t[1]))
            _, _, ref_plat_y, ref_plat_l, ref_plat_r = below_sorted[0]
            ref_plat_center_x = 0.5 * (ref_plat_l + ref_plat_r)

            self.opti.set_value(self.ref_center_x, ref_plat_center_x)
            self.opti.set_value(self.ref_top_y,   ref_plat_y)
            self.opti.set_value(self.ref_weight, 1e-3)  
        else:
            # Disable hover term
            self.opti.set_value(self.ref_center_x, 0.0)
            self.opti.set_value(self.ref_top_y,   0.0)
            self.opti.set_value(self.ref_weight,  0.0)

        ##Decide ONE platform to feed into potential wells

        chosen = None
        phase_str = ""

        if going_up:
            if above_cands:
                # Highest reachable above: sort by y then |dx|
                above_cands.sort(key=lambda t: (t[0], t[1]))  # smallest plat_y, then |dx|
                chosen = above_cands[0]
                phase_str = "UP (highest above)"
            elif below_cands:
                # Fallback: highest below (same criterion as hover ref)
                below_cands.sort(key=lambda t: (t[0], t[1]))  # smallest plat_y, then |dx|
                chosen = below_cands[0]
                phase_str = "UP (fallback highest below)"
            else:
                phase_str = "UP (no platforms)"
        else:
            if below_cands:
                # Falling: highest below
                below_cands.sort(key=lambda t: (t[0], t[1]))
                chosen = below_cands[0]
                phase_str = "DOWN (highest below)"
            else:
                phase_str = "DOWN (no platforms)"

        ## feed a bunch of dummies if none work
        if chosen is None:
            print(f"MPC: no vertically reachable platforms in {phase_str} phase.")
            dummy_y   = [1e9] * self.max_platforms
            dummy_l   = [1e9] * self.max_platforms
            dummy_r   = [-1e9] * self.max_platforms

            self.opti.set_value(self.plat_top_y,   dummy_y)
            self.opti.set_value(self.plat_x_left,  dummy_l)
            self.opti.set_value(self.plat_x_right, dummy_r)

        else:
            # Unpack chosen platform
            _, _, plat_y, x_left, x_right = chosen

            print(f"MPC chosen platform ({phase_str}): y={plat_y:.1f}, x=[{x_left:.1f}, {x_right:.1f}]")

            top_y_vals   = [plat_y]
            x_left_vals  = [x_left]
            x_right_vals = [x_right]

            # Pad remaining slots with dummies so only this one exerts attraction
            for _ in range(1, self.max_platforms):
                top_y_vals.append(1e9)
                x_left_vals.append(1e9)
                x_right_vals.append(-1e9)

            self.opti.set_value(self.plat_top_y,   top_y_vals)
            self.opti.set_value(self.plat_x_left,  x_left_vals)
            self.opti.set_value(self.plat_x_right, x_right_vals)

        ##Solve
        try:
            sol = self.opti.solve()
            u0 = float(sol.value(self.U[0, 0]))
            return u0
        except Exception as e:
            print("[MPC] solve failed:", e)
            return 0.0
