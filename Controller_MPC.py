from level import Level
import settings as config
import casadi as ca
from Dynamics import dynamics
from pygame.math import Vector2
import numpy as np


class PlatformState:
    def __init__(self, x, y, width, height, breakable):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.breakable = breakable


class MPCController:
    def __init__(
        self,
        level,
        mode="climb_fast",
        horizon=10,
        dt=config.DT,
        max_vel=Vector2(config.PLAYER_MAX_SPEED, 100),
    ):
        self.horizon = horizon
        self.dt = dt
        # mode: "climb_fast" or "energy_saving" (now used as point-collector)
        self.mode = mode
        self.levell = level
        self.model = dynamics(dt)
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(4, self.horizon + 1)  # [x, y, vx, vy]
        self.U = self.opti.variable(1, self.horizon)      # [u_x]

        # Parameters
        self.X0 = self.opti.parameter(4)
        self.X_target = self.opti.parameter(1)  # unused but kept
        self.max_platforms = config.MAX_PLAT
        self.plat_top_y   = self.opti.parameter(self.max_platforms)
        self.plat_x_left  = self.opti.parameter(self.max_platforms)
        self.plat_x_right = self.opti.parameter(self.max_platforms)

        # Reference-tracking term (center of chosen platform)
        self.ref_center_x = self.opti.parameter()
        self.ref_top_y    = self.opti.parameter()
        self.ref_weight   = self.opti.parameter()

        self.platforms = self.get_platforms()
        self.last_platform = None
        self.window_width = config.XWIN
        self.window_height = config.YWIN
        self.max_vel = max_vel

        # For point-collector mode, track which platforms were already hit
        self.visited_platform_ids = set()
        # Track the highest (minimum y) platform we've ever claimed points on
        self.highest_visited_y = None

        self._build_mpc()

    def _build_mpc(self):
        """
        Build MPC problem with:
        - platform-aware dynamics
        - potential field toward reachable platforms
        - mode-dependent cost
        - terminal cost and terminal constraints to approximate an invariant set
        """
        cost = 0.0

        # ---- Mode-dependent tuning parameters ----
        if self.mode == "energy_saving":
            # "Point-collector" mode:
            qy   = 0      # smaller height penalty
            qvx  = 1e-5       # global penalty on vx
            r    = 2e-3      # more expensive control
            w_plat = 1e-1    # platform potential weight
            qN_y = 0.0          # penalize ending low (large y) at terminal time

        else:
            # Default fast climber: directly wants high altitude
            qy   = 1e-2
            qvx  = 0.0       # no global penalty on vx
            r    = 1e-4      # cheap control
            w_plat = 1e-1
            qN_y = 5.0          # penalize ending low (large y) at terminal time


        u_max = 5.0

        # Length scales (in screen pixels) for the potential field
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

        # ### NEW: terminal weights ###
        # Stronger pull at the end of the horizon towards a "nice" state
        qN_ref_scale = 10.0 # scales the ref_weight at terminal time

        for k in range(self.horizon):
            Xk = self.X[:, k]
            Uk = self.U[:, k]

            # One-step prediction with platform-aware dynamics
            X_next = self.model.step_symbolic(
                Xk,
                Uk,
                self.plat_top_y,
                self.plat_x_left,
                self.plat_x_right,
            )
            self.opti.subject_to(self.X[:, k + 1] == X_next)

            x_next  = X_next[0]
            y_next  = X_next[1]
            vx_next = X_next[2]

            # --- Stage cost terms ---

            # 1) Penalty on being low (Pygame: larger y = lower on screen)
            cost += qy * (y_next**2)

            # 2) Control effort penalty
            cost += r * (Uk[0]**2)

            # 3) Optional global penalty on horizontal speed
            if qvx > 0:
                cost += qvx * (vx_next**2)

            # 4) Platform reference tracking (center/top), weight set from outside
            dx_ref    = x_next - self.ref_center_x
            dy_ref    = y_next - self.ref_top_y
            dist2_ref = dx_ref**2 + dy_ref**2
            cost += self.ref_weight * (dist2_ref + vx_next**2)

            # --- Platform potential field with phase-dependent reachability ---
            for i in range(self.max_platforms):
                plat_y = self.plat_top_y[i]
                plat_l = self.plat_x_left[i]
                plat_r = self.plat_x_right[i]

                center_x = 0.5 * (plat_l + plat_r)

                dx_plat = x_next - center_x
                dy_plat = y_next - plat_y

                # Smooth squared distance in (x, y) with length scales
                dist2   = (dx_plat / R_x)**2 + (dy_plat / R_y)**2
                attract = ca.exp(-dist2)  # biggest when near the platform

                # ---- Vertical reachability mask (based on initial y0, vy0) ----
                dy_up0   = y0 - plat_y
                dy_down0 = plat_y - y0

                cond_jump_reach = ca.logic_and(dy_up0 >= 0, dy_up0 <= max_jump_up_dyn)
                cond_fall_reach = ca.logic_and(dy_down0 >= 0, dy_down0 <= max_fall_down)

                cond_up_phase   = (vy0 < 0)
                cond_down_phase = (vy0 >= 0)

                cond_vert = ca.logic_or(
                    ca.logic_and(cond_up_phase,   cond_jump_reach),
                    ca.logic_and(cond_down_phase, cond_fall_reach),
                )

                # Only vertically reachable platforms create an attractive well
                well = ca.if_else(
                    cond_vert,
                    -w_plat * attract,  # reachable → reward being near
                    0.0,                # unreachable → no effect
                )
                cost += well

            # --- Constraints at step k (for k = 0..N-1) ---
            self.opti.subject_to(Uk[0] <= u_max)
            self.opti.subject_to(Uk[0] >= -u_max)

            self.opti.subject_to(self.X[2, k] <= self.max_vel.x)
            self.opti.subject_to(self.X[2, k] >= -self.max_vel.x)

        # ### NEW: terminal cost (approximate Lyapunov / invariant behavior) ###
        XN = self.X[:, self.horizon]
        xN = XN[0]
        yN = XN[1]
        vxN = XN[2]

        # Penalize ending low on the screen
        cost += qN_y * (yN**2)

        # Strongly pull terminal state to the platform center / top and small vx
        dxN_ref = xN - self.ref_center_x
        dyN_ref = yN - self.ref_top_y
        dist2N_ref = dxN_ref**2 + dyN_ref**2 + vxN**2

        # Use same ref_weight but scaled up at terminal time
        cost += (self.ref_weight * qN_ref_scale) * dist2N_ref

        # ### NEW: terminal constraints (terminal "set" X_f) ###
        # Ensure the terminal state lives in the same admissible region
        # as intermediate states; this approximates a control-invariant set.
        self.opti.subject_to(self.X[2, self.horizon] <= self.max_vel.x)
        self.opti.subject_to(self.X[2, self.horizon] >= -self.max_vel.x)
        self.opti.subject_to(self.X[0, self.horizon] >= 0)
        self.opti.subject_to(self.X[0, self.horizon] <= self.window_width)

        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.X0)

        # Objective
        self.opti.minimize(cost)

        # Solver
        self.opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    def get_platforms(self):
        platforms = []
        i = 1
        for p in self.levell.platforms:
            platforms.append(
                PlatformState(
                    x=p.rect.x,
                    y=p.rect.y,
                    width=p.rect.width,
                    height=p.rect.height,
                    breakable=p.breakable,
                )
            )
        return platforms

    # called from Player.onCollide when the agent lands on a platform
    def mark_platform_visited(self, platform):
        pid = id(platform)
        self.visited_platform_ids.add(pid)

        # Update the highest platform we've ever visited (smallest y)
        plat_y = float(platform.rect.y)
        if self.highest_visited_y is None or plat_y < self.highest_visited_y:
            self.highest_visited_y = plat_y

    def compute_control(self, state):
        # 1) Set initial state parameter - pygame coords
        X0_val = np.array(state).flatten()
        self.opti.set_value(self.X0, X0_val)

        # Unpack current player state
        px, py, vx, vy = map(float, state)

        g_abs = abs(config.GRAVITY) + 1e-6
        max_jump_up_dyn = (vy**2) / (2.0 * g_abs)
        max_fall_down = 800.0
        max_below_for_fallback = 1000.0

        going_up = (vy < 0.0)

        above_cands = []
        below_cands = []

        for p in self.levell.platforms:
            plat_y = float(p.rect.y)
            plat_l = float(p.rect.x)
            plat_r = float(p.rect.x + p.rect.width)
            center_x = 0.5 * (plat_l + plat_r)
            pid = id(p)

            # In "energy_saving" (point) mode, skip platforms already visited
            if self.mode == "energy_saving":
                # 1) Skip platforms we've already claimed points on
                if pid in self.visited_platform_ids:
                    continue

                # 2) Skip any platform that is *below* the highest platform we've ever visited
                #    (remember: larger y = lower on the screen)
                if self.highest_visited_y is not None and plat_y > self.highest_visited_y:
                    # This is below our best platform → never chase it
                    continue

            dy_up   = py - plat_y
            dy_down = plat_y - py
            dx      = center_x - px
            print(
                    "[CAND]",
                    "mode=", self.mode,
                    "going_up=", going_up,
                    "plat_y=", plat_y,
                    "dy_up=", dy_up,
                    "dy_down=", dy_down,
                    "dx=", dx,
                )
            # For sorting we use:
            #   t[0] = plat_y  (screen y position)
            #   t[1] = abs(dx) (horizontal distance)
            candidate = (plat_y, abs(dx), plat_y, plat_l, plat_r, pid)

            if going_up:
                # Upward phase: check reachable above first
                if dy_up > 0 and dy_up <= max_jump_up_dyn:
                    above_cands.append(candidate)
                    continue

                # Fallback: below but not too far
                if dy_down > 0 and dy_down <= max_below_for_fallback:
                    below_cands.append(candidate)
                    continue
            else:
                # Falling phase: platforms below within reachable fall window
                if dy_down > 0 and dy_down <= max_fall_down:
                    below_cands.append(candidate)

                # Decide chosen platform
        chosen = None
        phase_str = ""

        if self.mode == "energy_saving":
            if going_up:
                # ENERGY-SAVING, GOING UP:
                # consider ALL reachable platforms (above + below)
                # and pick the one with the "highest negative points"
                # = lowest on the screen => largest plat_y
                all_cands = above_cands + below_cands

                if all_cands:
                    # candidate = (plat_y, |dx|, plat_y, plat_l, plat_r, pid)
                    # sort by: largest y (lowest platform), then horizontally closest
                    all_cands.sort(key=lambda t: (-t[0], t[1]))
                    chosen = all_cands[0]
                    phase_str = "ENERGY-UP (max negative points, any platform)"
                else:
                    phase_str = "ENERGY-UP (no platforms)"
            else:
                # ENERGY-SAVING, GOING DOWN:
                # pick the lowest reachable platform below (largest y)
                # visited/too-low platforms are already filtered out above
                if below_cands:
                    below_cands.sort(key=lambda t: (-t[0], t[1]))
                    chosen = below_cands[0]
                    phase_str = "ENERGY-DOWN (lowest below, max negative points)"
                else:
                    phase_str = "ENERGY-DOWN (no platforms below)"
        else:
            # --- existing fast-climb logic, unchanged ---
            if going_up:
                if above_cands:
                    # FAST MODE: prefer highest reachable (smallest y), then closest horizontally
                    above_cands.sort(key=lambda t: (t[0], t[1]))
                    phase_str = "UP (highest above)"
                    chosen = above_cands[0]
                elif below_cands:
                    below_cands.sort(key=lambda t: (t[0], t[1]))
                    phase_str = "UP (fallback highest below)"
                    chosen = below_cands[0]
                else:
                    phase_str = "UP (no platforms)"
            else:
                if below_cands:
                    below_cands.sort(key=lambda t: (t[0], t[1]))
                    phase_str = "DOWN (highest below)"
                    chosen = below_cands[0]
                else:
                    phase_str = "DOWN (no platforms)"

        if chosen is None:
            print(f"MPC ({self.mode}): no vertically reachable platforms in {phase_str} phase.")
            dummy_y = [1e9] * self.max_platforms
            dummy_l = [1e9] * self.max_platforms
            dummy_r = [-1e9] * self.max_platforms

            self.opti.set_value(self.plat_top_y,   dummy_y)
            self.opti.set_value(self.plat_x_left,  dummy_l)
            self.opti.set_value(self.plat_x_right, dummy_r)

            # No specific platform reference
            self.opti.set_value(self.ref_center_x, 0.0)
            self.opti.set_value(self.ref_top_y,    0.0)
            self.opti.set_value(self.ref_weight,   0.0)
        else:
            # Unpack chosen platform
            # chosen = (plat_y, |dx|, plat_y, plat_l, plat_r, pid)
            plat_y = chosen[2]
            x_left = chosen[3]
            x_right = chosen[4]
            pid = chosen[5]

            print(
                f"MPC ({self.mode}) chosen platform ({phase_str}): "
                f"y={plat_y:.1f}, x=[{x_left:.1f}, {x_right:.1f}]"
            )

            # Only this one exerts attraction (rest are dummies)
            top_y_vals   = [plat_y]
            x_left_vals  = [x_left]
            x_right_vals = [x_right]

            for _ in range(1, self.max_platforms):
                top_y_vals.append(1e9)
                x_left_vals.append(1e9)
                x_right_vals.append(-1e9)

            self.opti.set_value(self.plat_top_y,   top_y_vals)
            self.opti.set_value(self.plat_x_left,  x_left_vals)
            self.opti.set_value(self.plat_x_right, x_right_vals)

            # --- Reference tracking setup ---
            plat_center_x = 0.5 * (x_left + x_right)
            self.opti.set_value(self.ref_center_x, plat_center_x)
            self.opti.set_value(self.ref_top_y,    plat_y)

            if self.mode == "energy_saving":
                # Point/collector mode: use a moderate tracking weight
                self.opti.set_value(self.ref_weight, 5e-4)
            else:
                # Fast mode: stronger tracking to hit chosen platform quickly
                self.opti.set_value(self.ref_weight, 1e-3)

        # Solve
        try:
            sol = self.opti.solve()
            u0 = float(sol.value(self.U[0, 0]))
            return u0
        except Exception as e:
            print(f"[MPC {self.mode}] solve failed:", e)
            return 0.0

