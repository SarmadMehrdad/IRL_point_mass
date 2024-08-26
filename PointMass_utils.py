import crocoddyl
import pinocchio
from IPython.display import HTML
import mim_solvers
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

def cost_residuals(x,u):
    if u is None:
        u = np.zeros(2)
    # Cost parameters # {Trans, Obs}
    r = np.zeros(2)
    target = np.array([10, 0, 0, 0])
    obs_center = np.array([5, 0])
    obs_rad = 2.0
    r[0] = np.linalg.norm(target - x)
    obs_d = np.linalg.norm(x[:2] - obs_center) - obs_rad
    r[1] = np.exp(-obs_d) # Trying a different cost form
    # if obs_d > obs_act:
    #     r[1] = 0
    # else:
    #     r[1] = np.linalg.norm(obs_d - obs_act)
    return r

def cum_feat(x, u, dt):
    cum_f = np.zeros(2)
    for X, U in zip(x[:-1],u):
        cum_f += cost_residuals(X,U)*dt
    cum_f += cost_residuals(x[-1], None)
    return cum_f

def traj_cost(x, u, w_run, w_term, dt):
    cost = 0
    for X, U in zip(x[:-1],u):
        cost += np.sum(w_run*cost_residuals(X,U))*dt
    cost += np.sum(w_term*cost_residuals(x[-1],None))
    return cost

def animatePointMass(xs, sleep=50, show=False):
    print("processing the animation ... ")
    mass_size = 1.0
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-5, 5))
    patch = plt.Circle((0, 0), radius=0.2, fc="b")
    obstacle = plt.Circle((5, 0), radius=2, fc="k")
    goal = plt.Rectangle((9.5,-0.5),1,1,fc="g") 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        ax.add_patch(goal)
        ax.add_patch(obstacle)
        ax.add_patch(patch)
        ax.set_aspect('equal', adjustable='box')
        time_text.set_text("")
        return patch, time_text

    def animate(i):
        x_pm = xs[i][0]
        y_pm = xs[i][1]
        patch.set_center((x_pm, y_pm))
        time = i * sleep / 1000.0
        time_text.set_text(f"time = {time:.1f} sec")
        return patch, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True
    )
    print("... processing done")
    if show:
        plt.show()
    return anim

