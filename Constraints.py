from pyomo.environ import *

window_width, window_height = self.window.get_size()

vx_max = dynamics.vx_max
vy_max = dynamics.vy_max
u_max = dynamics.u_max

model = ConcreteModel()

model.t = RangeSet(0, N)

model.x = Var(model.t, bounds=(0, window_width))
model.y = Var(model.t, bounds=(0, 2 * window_height))
model.vx = Var(model.t, bounds=(-vx_max, vx_max))
model.vy = Var(model.t, bounds=(-vy_max, vy_max))

model.u = Var(model.t, bounds=(-u_max, u_max))

model.x[0].fix(x0)
model.y[0].fix(y0)
model.vx[0].fix(vx0)
model.vy[0].fix(vy0)

def within_screen_y_rule(model, t):
    return 0 <= model.y[t] <= window_height

def within_screen_x_rule(model, t):
    return 0 <= model.x[t] <= window_width

def agent_alive(model, t):
    return model.y[t] <= 2 * window_height

def hit_platform_rule(model, t):
    return model.y[t] >= plat_y_top + 10000 * (model.vy[t] / vy_min)

#model.constraint1 = Constraint(rule=within_screen_x_rule)
#model.constraint2 = Constraint(rule=within_screen_y_rule)
#model.constraint3 = Constraint(rule=agent_alive)
model.constraint4 = Constraint(rule=hit_platform_rule)
