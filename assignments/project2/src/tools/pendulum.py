import numpy as np
import numpy.linalg as la

import lqr_discrete as lqr

class Pendulum:
    def __init__(self, T = 0.001, Q = np.eye(4), R = np.eye(1), x=np.array([[0.0],[0.0],[0.0],[0.0]]), u=np.array([[0.0]])):
        self.T = T
        self.A = np.array([
            [1,0,T,0],
            [0,1,0,-T],
            [0,-9.801*T,1,0],
            [0,21.582*T,0,0]#todo
        ])
        self.B = np.array([[0.0],[0.0],[0.1*T],[0.2*T]])
        self.Q = Q
        self.R = R

        self.K = None
        self.updataK()

        self.x = x
        self.u = u
    def setQ(self, Q):
        self.Q = Q

    def setR(self, R):
        self.R = R

    def init_state(self, x, u):
        self.x = x
        self.u = u

    def updataK(self, epochs=10):
        self.K = lqr.getSlideKN(self.A, self.B, self.Q, self.R, epochs, 4, 1)

    def step_in(self):
        '''
        update the state of the pendulum, in one step. The return value will be the new state of the pendulum.
        :return: state matrix of x
        '''
        self.u = -self.K @ self.u
        self.x = self.A @ self.x + self.B @ self.u
        return self.x.copy()