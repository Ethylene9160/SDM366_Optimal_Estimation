import numpy as np
# import sympy
# from sympy import *
import mujoco

import math
import matplotlib.pyplot as plt


class Droping:
    def __init__(self, T, m1=1.0, m2=1.0, l1=0.5, l2=0.5, theta1 = 0.0, theta2 = 0.0, g=9.81):
        self.T = T
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.x1 = theta1
        self.x2 = theta2
        self.x3 = 0 # theta1 dot
        self.x4 = 0 # theta2 dot

        self.M_theta = np.array([[self.l1 ** 2 * m1 + m2 * (self.l1 ** 2 + 2 * l1 * l2 * np.cos(self.x2) + l2 ** 2),
                                m2 * (l1 * l2 * np.cos(self.x2) + l2 ** 2)],
                               [m2 * (l1 * l2 * np.cos(self.x2) + l2 ** 2), l2 ** 2 * m2]])
        self.c_theta = np.array([[-m2 * l1 * l2 * np.sin(self.x2) * (2 * self.x3 * self.x4 + self.x4 ** 2)],
                               [m2 * l1 * l2 * self.x3 ** 2 * np.sin(self.x2)]])
        self.g_theta = np.array([[g * l1 * (m1 + m2) * np.cos(self.x1) + g * l2 * m2 * np.cos(self.x1 + self.x2)],
                               [g * l2 * m2 * np.cos(self.x1 + self.x2)]])

        self.tau_theta = np.array([[0], [0]])

        print('shape of M:', self.M_theta.shape)
        print('shape of c:', self.c_theta.shape)
        print('shape of g:', self.g_theta.shape)
        print('shape of tau:', self.tau_theta.shape)

    def _rectify_x(self):
        # self.x1 = self.x1 % (2*np.pi)
        # self.x2 = self.x2 % (2*np.pi)
        pass
    def _updateMatrix(self):
        # ensure x1 x2 to be scalar, not matrix or vector
        x2_scalar = float(self.x2)
        x1_scalar = float(self.x1)

        self.M_theta = np.array([
            [self.l1 ** 2 * self.m1 + self.m2 * (
                        self.l1 ** 2 + 2 * self.l1 * self.l2 * np.cos(x2_scalar) + self.l2 ** 2),
             self.m2 * (self.l1 * self.l2 * np.cos(x2_scalar) + self.l2 ** 2)],
            [self.m2 * (self.l1 * self.l2 * np.cos(x2_scalar) + self.l2 ** 2),
             self.l2 ** 2 * self.m2]
        ])

        self.c_theta = np.array([
            [-self.m2 * self.l1 * self.l2 * np.sin(x2_scalar) * (2 * self.x3 * self.x4 + self.x4 ** 2)],
            [self.m2 * self.l1 * self.l2 * self.x3 ** 2 * np.sin(x2_scalar)]
        ]).reshape((2, 1))

        self.g_theta = np.array([
            [self.g * self.l1 * (self.m1 + self.m2) * np.cos(x1_scalar) + self.g * self.l2 * self.m2 * np.cos(
                x1_scalar + x2_scalar)],
            [self.g * self.l2 * self.m2 * np.cos(x1_scalar + x2_scalar)]
        ])

        # calculate tau
        # angular_velocities = np.array([[self.x3], [self.x4]]).reshape(2, 1)
        # self.tau_theta = self.M_theta @ angular_velocities + self.c_theta + self.g_theta
        # print('shape of tau:', self.tau_theta.shape)
    def _dstep(self):
        # in descrete domain
        x1_new = self.x1 + self.x3 * self.T
        x2_new = self.x2 + self.x4 * self.T

        # a fast way to calculate M^{-1}*(tau-c-g)
        dtheta = np.linalg.solve(self.M_theta, (self.tau_theta - self.c_theta - self.g_theta))
        x3_new = self.x3 + self.T * dtheta[0]
        x4_new = self.x4 + self.T * dtheta[1]
        # x3_new = self._normalize_angle(x3_new)
        # x4_new = self._normalize_angle(x4_new)
        self.x1 = x1_new
        self.x2 = x2_new
        self.x3 = x3_new
        self.x4 = x4_new
        self._rectify_x()
        self._updateMatrix()

    def _normalize_angle(self, angle):
        """将角度归一化到 -pi 到 pi."""
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def forward(self, steps):
        x1s = []
        x2s = []
        x3s = []
        x4s = []
        for i in range(steps):
            self._dstep()
            x1s.append(float(self.x1))
            x2s.append(float(self.x2))
            x3s.append(float(self.x3))
            x4s.append(float(self.x4))
            # x3s.append(self.l1*math.cos(self.x1)+self.l2*math.cos(self.x1+self.x2))
            # x4s.append(self.l1*math.sin(self.x1)+self.l2*math.sin(self.x1+self.x2))
        return x1s, x2s, x3s, x4s

    def forwardNew(self, steps, T):
        x = np.array([[self.x1], [self.x2], [self.x3], [self.x4]])

    def forward_parameters(self, steps, T):
        theta1s = []
        theta2s = []
        d_theta1s = []
        d_theta2s = []
        d_d_theta1s = []
        d_d_theta2s = []
        tau1s = []
        tau2s = []
        for i in range(steps):
            self._dstep()
            theta1s.append(self.x1)
            theta2s.append(self.x2)
            d_theta1s.append(self.x3)
            d_theta2s.append(self.x4)
            d_d_theta1s.append(self.tau_theta[0])
            d_d_theta2s.append(self.tau_theta[1])
            tau1s.append(self.tau_theta[0])
            tau2s.append(self.tau_theta[1])
        return theta1s, theta2s, d_theta1s, d_theta2s, d_d_theta1s, d_d_theta2s, tau1s, tau2s

# todo
class LeastSquare:
    def __init__(self):
        self.H = None
        self.theta = None
        self.error = None
        self.flag = False

    def fit(self, y, H):
        self.H = H
        self.theta = np.linalg.inv(H.T @ H) @ H.T @ y
        self.error = np.sum((y - H @ self.theta) ** 2)
        self.flag = True
        return self.theta, self.error

    def predict(self, H):
        if not self.flag:
            raise ValueError('No model has been fitted yet.')
        return H @ self.theta

    def transformH(self, x):
        return np.array([[1, x, x**2, x**3]])


if __name__ == '__main__':

    # simulate 5 seconds
    steps = 5000
    T = 0.001
    time = np.arange(0, steps*T, T)


    d1 = Droping(T = T, theta1 = 0, theta2 = np.pi/2)
    # theta_1, theta_2, theta_1_dot, theta_2_dot
    x1s1,x2s1, x3s1, x4s1 = d1.forward(steps)


    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.plot(time, x1s1)
    plt.title('theta_1 varies with time')
    plt.subplot(222)
    plt.plot(time, x2s1)
    plt.title('theta_2 varies with time')
    plt.subplot(223)
    plt.plot(time, x3s1)
    plt.title('acc1 varies with time')
    plt.subplot(224)
    plt.plot(time, x4s1)
    plt.title('acc2 varies with time')
    plt.show()


    d2 = Droping(T = T, theta1 = 0, theta2 = -np.pi/2)
    x1s2, x2s2, x3s2, x4s2 = d2.forward(steps)
    # threhold = 19500
    # time = time[threhold:]
    # x1s2 = x1s2[threhold:]
    # x2s2 = x2s2[threhold:]
    # x3s2 = x3s2[threhold:]
    # x4s2 = x4s2[threhold:]
    plt.figure(figsize=(8,8))
    plt.subplot(221)
    plt.plot(time, x1s2)
    plt.title('\\theta_1 varies with time')
    plt.subplot(222)
    plt.plot(time, x2s2)
    plt.title('\\theta_2 varies with time')
    plt.subplot(223)
    plt.plot(time, x3s2)
    plt.title('\\dot \\theta_1 varies with time')
    plt.subplot(224)
    plt.plot(time, x4s2)
    plt.title('\\dot \\theta_2 varies with time')
    plt.show()



