import mujoco
import numpy as np
from . import Kalman

class VEstimator:
    def __init__(self, model):
        self.T = model.opt.timestep
        self.A = np.eye(3)
        self.B = self.T*np.eye(3)
        self.H = np.eye(3)
        self.Q = 0.1*np.eye(3)
        self.R = 0.1*np.eye(3)
        self.P = np.eye(3)
        self.Kalman = Kalman.Kalman(self.A, self.B, self.H, self.Q, self.R, self.P, np.zeros((3,1)))

    def update(self, x):
        return self.Kalman.update(x)

    def predict(self, u):
        return self.Kalman.predict(u)

