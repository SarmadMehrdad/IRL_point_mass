import mujoco
import cv2
import numpy as np
import yaml
import pickle
import pathlib
import pickle
import mujoco.viewer
from operator import index
import pinocchio as pin
from pinocchio.utils import *
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper, load_mujoco_model, get_robot_list
from numpy.linalg import norm, solve
import time
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
from matplotlib import cm, markers
from utils.path_recorder import PathRecorder

num_demo = 5
demo_size = 1000
cost_mu = []
cost_sigma = []
x0 = np.array([-1.5, -1.5])
goal = np.array([1.5, 1.5])
num_g = 10
# Randomize 10 Gaussians for cost
for i in range(num_g):
    cost_mu.append(np.random.rand(2)*2 -1)
    rand_vec = np.random.rand(2,2) + np.abs(np.random.rand(2,2))
    cost_sigma.append(rand_vec@rand_vec.T)

num_data = 100
x_range = np.linspace(-2.,2.,num_data)
y_range = np.linspace(-2.,2.,num_data)

cost = np.zeros(shape=(len(x_range), len(y_range)))

def interp(x, x_num):
    coordinates = [np.arange(0, x.shape[0])]
    interp_func = RegularGridInterpolator(coordinates, x, method='linear', bounds_error=False, fill_value=None)
    new_coordinates = [np.linspace(0, x.shape[0] - 1, x_num)]
    new_stacked_coordinates = np.meshgrid(*new_coordinates, indexing='ij')
    return interp_func(np.stack(new_stacked_coordinates, axis=-1))

for i in range(len(x_range)):
    for j in range(len(y_range)):
        for k in range(num_g):
            var = multivariate_normal(cost_mu[k], cost_sigma[k])
            cost[i,j] += var.pdf(np.array([x_range[i],y_range[j]]).T)
        
cost = cost/num_g
t = []
X = []
X_dot = []
X_ddot = []

for k in range(num_demo):
    path_recorder = PathRecorder()
    plt.contourf(x_range,y_range,cost,alpha=0.5, cmap=cm.jet)
    plt.plot(x0[0],x0[1],marker='X',color='black')
    plt.plot(goal[0],goal[1], marker='X', color='black')
    plt.show()
    t_k, path_k = path_recorder.get_path()
    t_k = interp(t_k,demo_size)
    path_k = interp(path_k,demo_size)
    t.append(t_k)
    X.append(path_k)
    dt = np.diff(t_k, axis=-1)[0]
    X_dot_k = np.gradient(path_k,dt,axis=0)
    X_ddot_k = np.gradient(X_dot_k,dt,axis=0)
    X_dot.append(X_dot_k)
    X_ddot.append(X_ddot_k)


X = np.array(X) # (Demo Number, Feature Values, Features [X, Y, Z, ...])
t = np.array(t) # (Demo Number, Time Recorded)
X_dot = np.array(X_dot) # (Demo Number, Feature Values, Features [X_dot, Y_dot, Z_dot, ...])
X_ddot = np.array(X_ddot) # (Demo Number, Feature Values, Features [X_ddot, Y_ddot, Z_ddot, ...])

print(X.shape)
print(t.shape)
print(X_dot.shape)
print(X_ddot.shape)

feature_length = X.shape[1]
data = {
    'time': t,
    'X': X,
    'X_dot': X_dot,
    'X_ddot': X_ddot,
    'x0': x0,
    'goal': goal,
    'num_g': num_g,
    'num_data': num_data,
    'feature_length': feature_length,
    'cost_mu': cost_mu,
    'cost_sigma': cost_sigma,
    'cost': cost
}

# with open('data.pkl', 'wb') as f:
#     pickle.dump(data, f)

# Let's code DMP
num_demo = 5
feature_length = X[0,:,0].shape[0]
bf_number = 50
K = np.diag(np.ones(feature_length))*5
D = 2 * np.sqrt(K)
conv_rate = 0.01
alpha = -np.log(conv_rate)

tau = t[:,-1]
dt = np.gradient(t,axis=-1)

s = np.exp(np.einsum('i,ij->ij',(-alpha/tau),t))[:,:,None]
s = np.tile(s, (1, 1, bf_number))
s = np.tile(s[0][None, None], (num_demo, 1, 1))
v = np.einsum('i,ijk->ijk',tau,X_dot)
v_dot = np.einsum('i,ijk->ijk',tau,X_ddot)
tau_v_dot = np.einsum('i,ijk->ijk', tau, v_dot)

K_inv = np.linalg.inv(K)
Dv = np.einsum('ij,djf->dif',D,v)
goal = np.tile(goal,(1,X.shape[1],1))

f_target = tau_v_dot + Dv + np.einsum('ij,djf->dif',K,(X-goal))

# Soliving for DMP Weights using DMP
ci= np.logspace(-3, 0, num=bf_number)
h = bf_number / (ci ** 2)
s_tile = np.tile(s[0][None], (num_demo, 1, 1))
ci_tile = np.tile(ci[None,None],(num_demo,feature_length,1))
psi_matrix = np.exp(-h * (s_tile - ci_tile) ** 2)
inv_sum_bfs = 1.0 / np.sum(psi_matrix, axis=-1)
bf_target = np.einsum('dlb,dl->dlb',psi_matrix*s_tile,inv_sum_bfs)

sol = np.linalg.lstsq(
    np.concatenate(f_target,axis=0),
    np.concatenate(bf_target,axis=0),
    rcond=None)

weights = sol[0]













s = np.exp((-alpha/tau) * t)
vx = tau*data['x_dot'].reshape(1,-1)
vy = tau*data['y_dot'].reshape(1,-1)
vx_dot = tau*data['x_ddot'].reshape(1,-1)
vy_dot = tau*data['y_ddot'].reshape(1,-1)
v_dot = np.concatenate((vx_dot,vy_dot),axis=0).T
v = np.concatenate((vx,vy),axis=0).T
X = np.concatenate((data['x'].reshape(1,-1),data['y'].reshape(1,-1)),axis=0).T
tau_vx_dot = tau*vx_dot
tau_vy_dot = tau*vy_dot
tau_v_dot = tau*v_dot
K_inv = np.linalg.inv(K)
Dv = D@v
goal = np.tile(goal,(X.shape[0],1))
f_target = tau_v_dot + Dv + K@(X - goal)

ci= np.logspace(-3, 0, num=bf_number)
hi = bf_number / (ci ** 2)

h = bf_number / (ci ** 2)
psi_matrix = np.zeros(shape=(feature_length,bf_number))
for i in range(bf_number):
    psi_matrix[:,i] = np.exp(-h[i] * (s - ci[i]) ** 2)

print(psi_matrix.shape)
print(s.shape)
inv_sum_bfs = 1.0 / np.sum(psi_matrix, axis=-1)
print(inv_sum_bfs.shape)
psi_matrix_s = np.zeros(shape=(feature_length,bf_number))
for i in range(feature_length):
    for j in range(bf_number):
        psi_matrix_s[i,j] = psi_matrix[i,j]*s[0,i]

inv_sum_bfs = 1 / np.sum(psi_matrix_s,axis=-1)
bf_target = np.einsum('ij,i->ij',psi_matrix_s,inv_sum_bfs)
sol = np.linalg.lstsq(bf_target, f_target,rcond=None)
theta = sol[0]