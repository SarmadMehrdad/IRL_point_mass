import crocoddyl
import pinocchio
from IPython.display import HTML
import mim_solvers
import numpy as np
import random
import pinocchio as pin
from matplotlib import animation
from matplotlib import pyplot as plt
from PointMass_utils import *
from PointMass_model import DifferentialActionModelPointMass
from pinocchio.robot_wrapper import RobotWrapper

# pm_model = pin.buildModelFromUrdf('IRL_point_mass/PM_model.urdf')
pm = RobotWrapper.BuildFromURDF('IRL_point_mass/PM_model.urdf', None, pin.JointModelFreeFlyer())
pm_model = pm.model
pm_data = pm.data
pm_model.gravity = pin.Motion.Zero()
# pm_data = pm_model.createData()
q_rand = pin.randomConfiguration(pm_model)
print(q_rand)
pin.forwardKinematics(pm_model, pm_data, q_rand, np.zeros(2))
pin.framesForwardKinematics(pm_model, pm_data, q_rand)
pin.updateFramePlacements(pm_model, pm_data)  
print(pm_model.jointPlacements.tolist())
