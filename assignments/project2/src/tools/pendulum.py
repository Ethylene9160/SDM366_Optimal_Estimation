import numpy as np
import numpy.linalg as la
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt

from . import lqr_discrete as lqr

# import control

class Pendulum:
    def __init__(self, T = 0.001, Q = np.eye(4), R = np.eye(1), x=np.array([[0.0],[0.0],[0.0],[0.0]]), u=np.array([[0.0]])):
        self.T = T
        self.A = np.array([
            [1,     0,      T,     0],
            [0,     1,      0,     -T],
            [0, -9.801*T,   1,     0],
            [0,-21.582*T,   0,     1]
        ])
        self.B = np.array([[0.0],[0.0],[0.1*T],[0.2*T]])

        self.Q = Q.copy()
        self.R = R.copy()

        print('shape of A: ', self.A.shape)
        print('shape of B: ', self.B.shape)
        print('shape of Q: ', self.Q.shape)
        print('shape of R: ', self.R.shape)
        self.K = None
        self.updataK()
        print('SHAPE of K: ', self.K.shape)
        self.x = x.copy()
        self.u = u.copy()

        self.zs = []
        self.thetas = []
    def setQ(self, Q):
        self.Q = Q

    def setR(self, R):
        self.R = R

    def init_state(self, x=np.array([[0.0],[0.0],[0.0],[0.0]]), u=np.array([[0.0]])):
        self.x = x.copy()
        self.u = u.copy()
        self.zs = []
        self.thetas = []

    def updataK(self, epochs=50000):
        # self.K = control.dlqr(self.A, self.B, self.Q, self.R)[0]
        self.K = lqr.getSlideKN(self.A, self.B, self.Q, self.R, epochs, 4, 1)

    def free_falling(self):
        self.x = self.A@self.x
        self.zs.append(self.x[0][0])
        self.thetas.append(float(np.pi)-self.x[1][0])
        return self.x.copy(), self.u.copy()

    def step_in(self):
        '''
        update the state of the pendulum, in one step. The return value will be the new state of the pendulum.
        :return: state matrix of x
        '''
        # print('shape of K: ', self.K.shape)
        # print('shape of u: ', self.u.shape)
        self.u = -self.K @ self.x
        self.x = self.A @ self.x + self.B @ self.u
        self.zs.append(self.x[0][0])
        self.thetas.append(float(np.pi)-self.x[1][0])
        return self.x.copy(), self.u.copy()

    def get_history(self):
        return np.array(self.zs), np.array(self.thetas)


def plot_robot_arm_dynamics(num_frames, zs, thetas, L1, L2, step=50):
    for i in range(0, len(zs), step):
        plt.figure(figsize=(8, 6))
        joint_x = L1/2+zs[i]
        joint_y = 0
        theta = thetas[i]-float(np.pi/2)
        end_effector_x = zs[i]+L2*np.cos(theta)
        end_effector_y = L2*np.sin(theta)

        plt.plot([-L1/2+zs[i], joint_x], [0, joint_y], 'ro-')
        plt.plot([zs[i], end_effector_x], [0, end_effector_y], 'bo-')
        plt.plot(end_effector_x, end_effector_y, 'go')
        plt.xlim([-L1 - L2 , L1 + L2 ])
        plt.ylim([-L1 - L2 , L1 + L2 ])
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.title(f'2R Robotic Arm Movement at t={i * 0.001:.2f} seconds')
        plt.grid(True)
        plt.show()
        clear_output(wait=True)