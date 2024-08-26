import crocoddyl
import pinocchio
from IPython.display import HTML
import mim_solvers
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

class Obstacle():
    def __init__(self, x, y, R):
        self.d = 0
        self.c = np.array([x, y])
        self.x = x
        self.y = y
        self.R = R

    def residual(self, X, U):
        self.d = np.linalg.norm(self.c - np.array([X[0], X[1]])) - self.R
        self.r = np.exp(-self.d)
        return self.r

class XReg():
    def __init__(self, nx, ref):
        self.nx = nx
        self.act = ref
        self.r = np.zeros(self.nx)

    def residual(self, X, U):
        if self.act is None:
            self.r = np.linalg.norm(X)
        else:
            self.r = np.linalg.norm(X - self.act)
        return self.r

class UReg():
    def __init__(self, nu, ref):
        self.nu = nu
        self.act = ref
        self.r = 0

    def residual(self, X, U):
        if U is None:
            U = np.zeros(self.nu)
        if self.act is None:
            self.r = np.linalg.norm(U)
        else:
            self.r = np.linalg.norm(U - self.act)
        return self.r

class Costs():
    def __init__(self):
        self.nr = 0
        self.costs = []
        self.w = []
        self.d = []
        self.r = []

    def add_cost(self, cost_model):
        self.nr += 1
        self.costs.append(cost_model)
        # self.w.append(w)

    def residuals(self, X, U):
        self.r = np.zeros(self.nr)
        for i, cost_m in enumerate(self.costs):
            self.r[i] = cost_m.residual(X, U)
        return self.r
        
    def cum_feat(self, x, u, dt):
        cum_f = np.zeros(self.nr)
        for X, U in zip(x[:-1],u):
            cum_f += self.residuals(X,U)*dt
        cum_f += self.residuals(x[-1], None)
        return cum_f
    
    def traj_cost(self, x, u, w_run, w_term, dt):
        cost = 0
        for X, U in zip(x[:-1],u):
            cost += np.sum(w_run*self.residuals(X,U))*dt
        cost += np.sum(w_term*self.residuals(x[-1],None))
        return cost

def normalize(x):
    return x/np.max(x)

def animatePointMass(xs, obstacles, target, sleep=50, show=False):
    print("processing the animation ... ")
    mass_size = 1.0
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    patch = plt.Circle((0, 0), radius=0.2, fc="b")
    obs_set = []
    for obs in obstacles:
        obs_set.append(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k"))

    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g") 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        ax.add_patch(goal)
        for obs in obs_set:
            ax.add_patch(obs)
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

def plot_results(x_opt, x_nopt, x_irl, obstacles, target):
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g") 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.add_patch(goal)
    for obs in obstacles:
        ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k"))
    ax.set_aspect('equal', adjustable='box')
    time_text.set_text("")
    plt.plot(x_opt[:,0],x_opt[:,1], 'k:', label='Optimal')
    for x in x_nopt[:-1]:
        plt.plot(x[:,0],x[:,1], 'r:', label='_nolegend_')
    plt.plot(x_nopt[-1][:,0],x_nopt[-1][:,1], 'r:', label='Non-Optimal')
    plt.plot(x_irl[:,0], x_irl[:,1], 'b-', label='IRL')
    plt.legend()
    plt.show()