from calendar import c
import crocoddyl
import pinocchio
from IPython.display import HTML
import mim_solvers
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

class Obstacle():
    def __init__(self, x, y, R, act, name):
        self.d = 0.0
        self.act = act
        self.c = np.array([x, y])
        self.x = x
        self.y = y
        self.r = 0.0
        self.R = R
        self.name = name

    def residual(self, X, U):
        self.d = np.linalg.norm(self.c - np.array([X[0], X[1]])) - self.R
        if self.d > self.act:
            self.r = 0.0
        else:
            self.r = np.linalg.norm(self.d - self.act)
            # self.r = np.exp(-self.d) # Exponential Weighting for Obstacle
        return self.r

class QReg():
    def __init__(self, nq, ref, name):
        self.nq = nq
        self.ref = ref
        self.r = 0.0
        self.name = name

    def residual(self, X, U):
        if self.ref is None:
            self.r = np.linalg.norm(X[:self.nq])
        else:
            self.r = np.linalg.norm(X[:self.nq] - self.ref)
        return self.r

class XReg():
    def __init__(self, nx, ref, name):
        self.nx = nx
        self.ref = ref
        self.r = 0.0
        self.name = name

    def residual(self, X, U):
        if self.ref is None:
            self.r = np.linalg.norm(X)
        else:
            self.r = np.linalg.norm(X - self.ref)
        return self.r

class UReg():
    def __init__(self, nu, ref, name):
        self.nu = nu
        self.ref = ref
        self.r = 0
        self.name = name

    def residual(self, X, U):
        if U is None:
            U = np.zeros(self.nu)
        if self.ref is None:
            self.r = np.linalg.norm(U)
        else:
            self.r = np.linalg.norm(U - self.ref)
        return self.r

class Costs():
    def __init__(self):
        self.nr = 0
        self.costs = []
        self.w = []
        self.d = []
        self.r = None
        self.names = []

    def add_cost(self, cost_model):
        self.nr += 1
        self.costs.append(cost_model)
        self.names.append(cost_model.name)

    def residuals(self, X, U):
        self.r = np.zeros(self.nr)
        for i, cost_m in enumerate(self.costs):
            self.r[i] = cost_m.residual(X, U)
        return self.r
        
    def cum_feat(self, x, u, dt):
        cum_f = np.zeros(self.nr)
        for X, U in zip(x[:-1],u):
            cum_f += 0.5*(self.residuals(X,U)**2)
        cum_f += 0.5*(self.residuals(x[-1], None)**2)
        return cum_f
    
    def traj_cost(self, x, u, w_run, w_term, dt):
        cost = 0
        for X, U in zip(x[:-1],u):
            cost += 0.5*np.sum(w_run*(self.residuals(X,U)**2))*dt
        cost += 0.5*np.sum(w_term*(self.residuals(x[-1],None)**2))
        return cost
    
    def traj_cost_and_feat(self, x, u, w_run, w_term, dt):
        cost = 0
        cum_f = np.zeros(self.nr)
        for X, U in zip(x[:-1],u):
            res = self.residuals(X,U)**2
            f = 0.5*(res)
            cum_f += f
            cost += np.sum(w_run*f)*dt
        res = self.residuals(x[-1],None)**2
        f = 0.5*(res)
        cum_f += f
        cost += np.sum(w_term*f)
        return cost, cum_f

    def traj_cost_and_feat_modified(self, x, u, w_run, w_term, dt, mean=None, cov_inv_sqrt=None):
        if cov_inv_sqrt is None:
            cov_inv_sqrt = np.eye(self.nr)
        if mean is None:
            mean = np.zeros(self.nr)
        cost = 0
        cum_f = np.zeros(self.nr)
        for X, U in zip(x[:-1],u):
            res = self.residuals(X,U)**2
            res_mod = res[:,None] - mean[:,None]; res_mod = cov_inv_sqrt @ res_mod; res_mod += mean[:,None]
            f = 0.5*(np.squeeze(res_mod))
            cum_f += f
            cost += np.sum(w_run*f)*dt
        res = self.residuals(x[-1],None)
        res_mod = res[:,None] - mean[:,None]; res_mod = cov_inv_sqrt @ res_mod; res_mod += mean[:,None]
        f = 0.5*(np.squeeze(res_mod))
        cum_f += f
        cost += np.sum(w_term*f)
        return cost, cum_f



def normalize(x):
    if np.max(x) != 0.0:
        return x/np.max(x)
    else:
        return x

def set_merge(x,y):
    z = []
    for x_ in x:
        z.append(x_)
    for y_ in y:
        z.append(y_)
    return z

def animatePointMass(xs, obstacles, target, sleep=50, show=False):
    print("processing the animation ... ")
    mass_size = 1.0
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    patch = plt.Circle((0, 0), radius=0.2, fc="b")
    obs_set = []
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        obs_set.append(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            obs_set.append(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")

    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
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

def animateTraj(xs_opt, xs_set, obstacles, target, set_size, sleep=1000, animType=1):
    print("processing the animation ... ")
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    obs_set = []
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        obs_set.append(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            obs_set.append(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")

    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        ax.add_patch(goal)
        for obs in obs_set:
            ax.add_patch(obs)
        ax.set_aspect('equal', adjustable='box')
        time_text.set_text("")
        ax.plot(xs_opt[:,0],xs_opt[:,1],'g:')
        return ax, time_text

    def animate1(i):
        ax.cla()
        init()
        X = xs_set[:i]
        alpha_value = 1
        for x_ in X:
            ax.plot(x_[:,0], x_[:,1],color='r',alpha=(0.4*alpha_value/(len(X))))
            alpha_value += 1
        time = i * sleep / 1000.0
        time_text.set_text(f"time = {time:.1f} sec")
        return ax, time_text

    def animate2(i):
        ax.cla()
        init()
        X = xs_set[i]
        ax.plot(X[:,0], X[:,1],color='r',alpha=(0.5))
        time = i * sleep / 1000.0
        time_text.set_text(f"time = {time:.1f} sec")
        return ax, time_text
    
    def animate3(i):
        ax.cla()
        init()
        if i < set_size:
            X = xs_set[:i]
        else:
            X = xs_set[i-set_size:i]
        alpha_value = 1
        for x_ in X[:-1]:
            ax.plot(x_[:,0], x_[:,1],color='r',alpha=(0.4*alpha_value/(len(X))))
            alpha_value += 1
        x_ = X[-1]; ax.plot(x_[:,0], x_[:,1],color='b',alpha=(0.4*alpha_value/(len(X))))
        time = i * sleep / 1000.0
        time_text.set_text(f"time = {time:.1f} sec")
        return ax, time_text
        
    if animType == 1:
        anim = animation.FuncAnimation(
            fig, animate1, init_func=init, frames=len(xs_set), interval=sleep, blit=False
        )
    elif animType == 2:
        anim = animation.FuncAnimation(
            fig, animate2, init_func=init, frames=len(xs_set), interval=sleep, blit=False
        )
    elif animType == 3:
        anim = animation.FuncAnimation(
            fig, animate3, init_func=init, frames=len(xs_set), interval=sleep, blit=False
        )
        
    print("... processing done")
    return anim

def plot_results(x_opt, x_nopt, x_irl, obstacles, target):
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.add_patch(goal)
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    ax.set_aspect('equal', adjustable='box')
    time_text.set_text("")
    for x in x_nopt[:-1]:
        plt.plot(x[:,0],x[:,1], 'r', alpha=0.2, label='_nolegend_')
    plt.plot(x_nopt[-1][:,0],x_nopt[-1][:,1], 'r', alpha=0.3, label='Non-Optimal')
    plt.plot(x_opt[:,0],x_opt[:,1], 'g', label='Optimal')
    plt.plot(x_irl[:,0], x_irl[:,1], 'b-', label='IRL')
    plt.legend()
    plt.show()

def plot_1_traj(x, obstacles, target, label='', linemap = 'k:'):
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.add_patch(goal)
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    ax.set_aspect('equal', adjustable='box')
    time_text.set_text("")
    plt.plot(x[:,0],x[:,1], linemap, label=label)
    plt.legend()
    plt.show()

def plot_1_set(x, obstacles, target, label='', linemap_traj = 'b', linemap_set='r:', traj_alpha = 0.5):
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.add_patch(goal)
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    ax.set_aspect('equal', adjustable='box')
    time_text.set_text("")
    x_traj = x[0]
    plt.plot(x_traj[:,0],x_traj[:,1], linemap_traj, label=label, alpha=traj_alpha)
    for x_set in x[1:]:
        plt.plot(x_set[:,0],x_set[:,1], linemap_set, label='_nolegend_', alpha=traj_alpha)
    plt.legend()
    plt.show()

def plot_1_multiset(x, x_set, obstacles, target, label='', linemap_traj = 'g:', linemap_set='r:'):
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.add_patch(goal)
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    ax.set_aspect('equal', adjustable='box')
    time_text.set_text("")
    for i, X in enumerate(x):
        plt.plot(X[:,0],X[:,1], linemap_traj, label=label)
        for Xs in x_set[i]:
            plt.plot(Xs[:,0],Xs[:,1], linemap_set, label='_nolegend_')
    plt.legend()
    plt.show()

def check_collision(x, obs_set):
    collision = False
    for x_ in x:
        for obs in obs_set:
            d = np.linalg.norm(x_[:2] - np.array([obs.x, obs.y]))
            if d < obs.R:
                collision = True
    return collision

def plot_tested_model(xs_set, obstacles, target):
    num_col = 0
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 12), ylim=(-2, 12))
    goal = plt.Rectangle((target[0]-0.5,target[1]-0.5),1,1,fc="g", alpha=0.7) 
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.add_patch(goal)
    if len(obstacles) ==1:
        ax.add_patch(plt.Circle((obstacles[0].x, obstacles[0].y), radius=obstacles[0].R, fc="k", alpha=0.5))
        ax.text(obstacles[0].x, obstacles[0].y, "1",bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    else:
        for i, obs in enumerate(obstacles):
            ax.add_patch(plt.Circle((obs.x, obs.y), radius=obs.R, fc="k", alpha=0.5))
            ax.text(obs.x, obs.y, str(i+1),bbox={"boxstyle" : "circle", "color":"grey"},ha="center",va="center")
    ax.set_aspect('equal', adjustable='box')
    time_text.set_text("")
    for xs in xs_set:
        col = check_collision(xs, obstacles)
        if col:
            color = 'r'
            num_col += 1
        else:
            color = 'b'
        plt.plot(xs[:,0],xs[:,1], color=color, alpha=0.3)
    plt.show()
    print('Number of Collisions: {}'.format(num_col))
        

def distributions(cost_set, x_set, u_set, w_run, w_term, dt):
    P = np.zeros(len(x_set))
    costs = np.zeros(len(u_set))
    for i, (X,U) in enumerate(zip(x_set, u_set)):
        costs[i] = cost_set.traj_cost(X, U, w_run, w_term, dt)
    den = 0.0
    for i, cost in enumerate(costs):
        P[i] = np.exp(-cost)
        den += P[i]
    
    P /= den
    return P

