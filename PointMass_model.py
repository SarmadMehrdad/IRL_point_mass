import crocoddyl
import numpy as np
from PointMass_utils import Costs

class DifferentialActionModelPointMass(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, cost_model, w):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), nu = 2, nr = cost_model.nr
        )  
        # nu = 2 {Fx, Fy} 
        # nr = 2 {Trans, Obs}
        self.unone = np.zeros(self.nu)
        self.m = 1.0 # Dynamics (so to say !!)
        self.cost_model = cost_model
        self.costWeights = w.copy()

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        # Getting the state and control variables
        X, Y, Xdot, Ydot = x[0], x[1], x[2], x[3]
        fx = u[0]
        fy = u[1]

        # Shortname for system parameters
        m = self.m

        # Defining the equation of motions
        Xddot = fx / m
        Yddot = fy / m
        data.xout = np.matrix([Xddot, Yddot]).T
        data.r = self.cost_model.residuals(x, u) 
        data.cost = 0.5 * np.sum(self.costWeights * np.asarray(data.r) ** 2)
        # data.cost = np.sum(self.costWeights * np.asarray(data.r))