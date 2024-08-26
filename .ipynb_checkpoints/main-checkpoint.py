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

import matplotlib.pyplot as plt
from matplotlib import cm, markers
from utils.path_recorder import PathRecorder

cost_mu = []
cost_sigma = []
x0 = np.array([-1.5, -1.5])
goal = np.array([1.5, 1.5])
num_g = 10
# Randomize 10 Gaussians for cost
for i in range(num_g):
    cost_mu.append(np.random.rand(2)*2 -1)
    rand_vec = np.random.rand(2,2) + 1
    cost_sigma.append(rand_vec@rand_vec.T)

num_data = 10
x_range = np.linspace(-2.,2.,num_data)
y_range = np.linspace(-2.,2.,num_data)

cost = np.zeros(shape=(len(x_range), len(y_range)))

for i in range(len(x_range)):
    for j in range(len(y_range)):
        for k in range(num_g):
            var = multivariate_normal(cost_mu[k], cost_sigma[k])
            cost[i,j] += var.pdf(np.array([x_range[i],y_range[j]]).T)
        
cost = cost/num_g

path_recorder = PathRecorder()
plt.contourf(x_range,y_range,cost,alpha=0.5, cmap=cm.jet)
plt.plot(x0[0],x0[1],marker='X',color='black')
plt.plot(goal[0],goal[1], marker='X', color='black')
plt.show()

t, path = path_recorder.get_path()
# print("Recorded path:", path)
# print("Time", t)
# print("Data point num:", path.shape)
# print("Time point num:", t.shape)

x_dot = np.gradient(path[:,0],t)
y_dot = np.gradient(path[:,1],t)
# print("VX Shape:", vx.shape, "VY Shape: ", vy.shape)
x_ddot = np.gradient(x_dot,t)
y_ddot = np.gradient(y_dot,t)
# print("AX Shape:", ax.shape, "AY Shape: ", ay.shape)

# print(t.shape,path[:,0].shape,path[:,1].shape,vx.shape,vy.shape,ax.shape,ay.shape)

feature_length = path.shape[0]
# feature_length = 100
data = {
    'time': t,
    'x': path[:,0],
    'y': path[:,1],
    'x_dot': x_dot,
    'y_dot': y_dot,
    'x_ddot': x_ddot,
    'y_ddot': y_ddot
}
# print(data)

# plt.subplot(3,2,1)
# plt.plot(data['time'],data['x'])
# plt.subplot(3,2,2)
# plt.plot(data['time'],data['y'])
# plt.subplot(3,2,3)
# plt.plot(data['time'],data['x_dot'])
# plt.subplot(3,2,4)
# plt.plot(data['time'],data['y_dot'])
# plt.subplot(3,2,5)
# plt.plot(data['time'],data['x_ddot'])
# plt.subplot(3,2,6)
# plt.plot(data['time'],data['y_ddot'])

# plt.show()

# Let's code DMP
bf_number = 50
K = np.diag(np.ones(feature_length))*5
D = 2 * np.sqrt(K)
conv_rate = 0.01
alpha = -np.log(conv_rate)

tau = data['time'][-1].reshape(1,-1)
dt = np.gradient(data['time'],axis=-1)

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


psi_matrix_s = np.einsum('ij,i->ij',psi_matrix, s)
print(psi_matrix_s.shape)
# bf_target = np.einsum('tdlb,tdl->tdlb', psi_matrix * s, inv_sum_bfs)