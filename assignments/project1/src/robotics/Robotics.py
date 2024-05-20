import numpy as np
# import sympy
# from sympy import *
import mujoco
import control

import math
import matplotlib.pyplot as plt

# x1 = theta_1
# x2 = theta_2
# x3 = theta_1_dot
# x4 = theta_2_dot

# x1 = Symbol('x1')
# x2 = Symbol('x2')
# x3 = Symbol('x3')
# x4 = Symbol('x4')
#
# x1_dot = Symbol('\dot x_1')
# x2_dot = Symbol('\dot x_2')
# x3_dot = Symbol('\dot x_3')
# x4_dot = Symbol('\dot x_4')

# l1 = Symbol('l1')
# l2 = Symbol('l2')
# m1 = Symbol('m1')
# m2 = Symbol('m2')
# theta1 = Symbol('theta1')
# theta2 = Symbol('theta2')
# g = Symbol('g')
# tau1 = Symbol('tau1')
# tau2 = Symbol('tau2')


# M = Matrix([[m1*l1**2 + m2*(l1**2 + 2*l1*l2*cos(x2) + l2**2), m2*(l1*l2*cos(x2) + l2**2)],
#         [m2*(l1*l2*cos(x2) + l2**2), m2*l2**2]])
# c = Matrix([[-m2*l1*l2*sin(x2)*(2*x1_dot*x2_dot + x2_dot**2)],
#         [m2*l1*l2*x1_dot**2*sin(x2)]])
# g_theta = Matrix([[(m1+m2)*l1*g*cos(x1)+m2*g*l2*cos(x1+x2)],
#         [m2*g*l2*cos(x1+x2)]])
#
# tau = Matrix([[tau1],
#         [tau2]])

# Calculate the inverse of matrix M
# M_inv = M.inv()

# Generate the LaTeX representation of the matrix inverse
# M_inv_latex = latex(M_inv)
# print(M_inv_latex)
#
# print(latex(c))
# print(latex(g_theta))
# print(latex(tau))

# calculate state function

# resultMatrix = M_inv * (tau - c - g_theta)
# print('result: ')
# print(latex(resultMatrix))
#
# print('M')
# print(M)
# print('c')
# print(c)
# print('g_theta')
# print(g_theta)
# print('tau')
# print(tau)
# flag = 1
class Droping:
    def __init__(self, T, m1=1.0, m2=1.0, l1=0.5, l2=0.5, theta1 = 0.0, theta2 = 0.0, g=9.81):
        self.flag = 1
        self.T = T
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.x1 = theta1
        self.x2 = theta2
        self.x3 = 0
        self.x4 = 0
        # self.M_theta = Matrix([[self.l1**2*m1 + m2*(self.l1**2 + 2*l1*l2*cos(self.x2) + l2**2),
        #                         m2*(l1*l2*cos(self.x2) + l2**2)],
        #                        [m2*(l1*l2*cos(self.x2) + l2**2), l2**2*m2]])
        # self.c_theta = Matrix([[-m2*l1*l2*sin(self.x2)*(2*self.x3*self.x4 + self.x4**2)],
        # [m2*l1*l2*self.x3**2*sin(self.x2)]])
        # self.g_theta = Matrix([[g*l1*(m1 + m2)*cos(self.x1) + g*l2*m2*cos(self.x1 + self.x2)], [g*l2*m2*cos(self.x1 + self.x2)]])
        # self.tau_theta = self.M_theta*Matrix([[self.x3], [self.x4]]) + self.c_theta + self.g_theta #+Matrix([[2.5],[0]])

        self.M_theta = np.array([[self.l1 ** 2 * m1 + m2 * (self.l1 ** 2 + 2 * l1 * l2 * np.cos(self.x2) + l2 ** 2),
                                m2 * (l1 * l2 * np.cos(self.x2) + l2 ** 2)],
                               [m2 * (l1 * l2 * np.cos(self.x2) + l2 ** 2), l2 ** 2 * m2]])
        self.c_theta = np.array([[-m2 * l1 * l2 * np.sin(self.x2) * (2 * self.x3 * self.x4 + self.x4 ** 2)],
                               [m2 * l1 * l2 * self.x3 ** 2 * np.sin(self.x2)]])
        self.g_theta = np.array([[g * l1 * (m1 + m2) * np.cos(self.x1) + g * l2 * m2 * np.cos(self.x1 + self.x2)],
                               [g * l2 * m2 * np.cos(self.x1 + self.x2)]])
        # self.tau_theta = self.M_theta @ np.array(
        #     [[self.x3], [self.x4]]) + self.c_theta + self.g_theta  +np.array([[2.5],[0]])
        self.tau_theta = np.array([[0], [0]])

        print('shape of M:', self.M_theta.shape)
        print('shape of c:', self.c_theta.shape)
        print('shape of g:', self.g_theta.shape)
        print('shape of tau:', self.tau_theta.shape)

        # self.A = Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0,0,0,0],[0,0,0,0]])
        # temp = self.M_theta.inv() * (self.tau_theta - self.c_theta - self.g_theta)
        # print(latex(temp))
        # self.A = self.A.row_insert(2, temp)
        # print(latex(self.A))
        # x_dots = np.linalg.solve(self.M_theta, (self.tau_theta - self.c_theta - self.g_theta))
        # self.x3 = x_dots[0]
        # self.x4 = x_dots[1]
    def _rectify_x(self):
        # self.x1 = self.x1 % (2*np.pi)
        # self.x2 = self.x2 % (2*np.pi)
        pass
    def _updateMatrix(self):
        # M_theta_new = Matrix([[self.l1 ** 2 * self.m1 + self.m2 * (self.l1 ** 2 + 2 * self.l1 * self.l2 * cos(self.x2) + self.l2 ** 2),
        #                         self.m2 * (self.l1 * self.l2 * cos(self.x2) + self.l2 ** 2)],
        #                        [self.m2 * (self.l1 * self.l2 * cos(self.x2) + self.l2 ** 2),
        #                         self.l2 ** 2 * self.m2]])
        # c_theta_new = Matrix([[-self.m2 * self.l1 * self.l2 * sin(self.x2) * (2 * self.x3 * self.x4 + self.x4 ** 2)],
        #                        [self.m2 * self.l1 * self.l2 * self.x3 ** 2 * sin(self.x2)]])
        # g_theta_new = Matrix([[self.g * self.l1 * (self.m1 + self.m2) * cos(self.x1) + self.g * self.l2 * self.m2 * cos(self.x1 + self.x2)],
        #                        [self.g * self.l2 * self.m2 * cos(self.x1 + self.x2)]])
        # tau_theta_new = M_theta_new*Matrix([[self.x3], [self.x4]]) + c_theta_new + g_theta_new

        # 确保 x1 和 x2 是标量
        x2_scalar = float(self.x2)
        x1_scalar = float(self.x1)

        # 更新 M, c, g, tau 矩阵
        # print("l1^2 * m1:", self.l1 ** 2 * self.m1)
        # print("m2 * (l1^2 + 2 * l1 * l2 * cos(x2) + l2^2):",
        #       self.m2 * (self.l1 ** 2 + 2 * self.l1 * self.l2 * np.cos(x2_scalar) + self.l2 ** 2))
        # print("m2 * (l1 * l2 * cos(x2) + l2^2):", self.m2 * (self.l1 * self.l2 * np.cos(x2_scalar) + self.l2 ** 2))
        # print("l2^2 * m2:", self.l2 ** 2 * self.m2)

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

        # print('shape of M:', self.M_theta.shape)
        # print('shape of c:', self.c_theta.shape)
        # print('shape of g:', self.g_theta.shape)
        dthetas = np.linalg.solve(self.M_theta, (self.tau_theta - self.c_theta - self.g_theta))
        # if(self.flag % 500 == 0):
        #     self.flag = self.flag + 1
        #     print('d d thetas: ', dthetas)

        # 计算力矩 tau
        angular_velocities = np.array([[float(dthetas[0])], [float(dthetas[1])]]).reshape(2, 1)
        self.tau_theta = self.M_theta @ angular_velocities + self.c_theta + self.g_theta

        # print('shape of tau:', self.tau_theta.shape)
    def _dstep(self):

        # x = np.array([[self.x1], [self.x2], [self.x3], [self.x4]])
        # M = self.M_theta



        x1_new = self.x1 + self.x3 * self.T
        x2_new = self.x2 + self.x4 * self.T
        # temp = np.array([[self.T,0],[0,self.T]])*self.M_theta.inv() * (self.tau_theta - self.c_theta - self.g_theta)
        # x3_new = temp[0]+self.x3
        # x4_new = temp[1]+self.x4

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
        return dtheta[0], dtheta[1]

    def _normalize_angle(self, angle):
        """将角度归一化到 -pi 到 pi."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _dstep0(self):
        # 使用前向欧拉方法更新状态
        # 如果使用RK4，这里需要用RK4的步骤替代
        x1_new = self._normalize_angle(self.x1 + self.x3 * self.T)
        x2_new = self._normalize_angle(self.x2 + self.x4 * self.T)

        # 为了避免求逆，这里我们解一个线性方程 M_theta * dtheta = tau_theta - c_theta - g_theta
        dtheta = np.linalg.solve(self.M_theta, (self.tau_theta - self.c_theta - self.g_theta))

        x3_new = self.x3 + self.T * dtheta[0]
        x4_new = self.x4 + self.T * dtheta[1]

        # self.x1 = self._normalize_angle(x1_new)
        # self.x2 = self._normalize_angle(x2_new)
        self.x1 = x1_new
        self.x2 = x2_new
        self.x3 = x3_new
        self.x4 = x4_new

        self._updateMatrix()

    def forward(self, steps):
        x1s = []
        x2s = []
        x3s = []
        x4s = []
        for i in range(steps):
            self._dstep()
            x1s.append(float(self.x1))
            x2s.append(float(self.x2))
            # x3s.append(float(self.x3))
            # x4s.append(float(self.x4))
            x3s.append(self.l1*math.cos(self.x1)+self.l2*math.cos(self.x1+self.x2))
            x4s.append(self.l1*math.sin(self.x1)+self.l2*math.sin(self.x1+self.x2))
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
            # dt0, dt1 = self._dstep()
            # # theta1 theta2
            # theta1s.append(float(self.x1))
            # theta2s.append(float(self.x2))
            # # theta1 hat, theta2 hat
            # d_theta1s.append(float(self.x3))
            # d_theta2s.append(float(self.x4))
            # # d_d_theta1s.append(float(self.tau_theta[0]))
            # # d_d_theta2s.append(float(self.tau_theta[1]))
            # d_d_theta1s.append(dt0)
            # d_d_theta2s.append(dt1)

            # x = np.array([[self.x1], [self.x2], [self.x3], [self.x4]])
            # M = self.M_theta

            x1_new = self.x1 + self.x3 * self.T
            x2_new = self.x2 + self.x4 * self.T
            # temp = np.array([[self.T,0],[0,self.T]])*self.M_theta.inv() * (self.tau_theta - self.c_theta - self.g_theta)
            # x3_new = temp[0]+self.x3
            # x4_new = temp[1]+self.x4
            theta1s.append(float(x1_new))
            theta2s.append(float(x2_new))
            dtheta = np.linalg.solve(self.M_theta, (self.tau_theta - self.c_theta - self.g_theta))

            x3_new = self.x3 + self.T * dtheta[0]
            x4_new = self.x4 + self.T * dtheta[1]
            d_theta1s.append(float(x3_new))
            d_theta2s.append(float(x4_new))

            d_d_theta1s.append(float(dtheta[0]))
            d_d_theta2s.append(float(dtheta[1]))
            # x3_new = self._normalize_angle(x3_new)
            # x4_new = self._normalize_angle(x4_new)
            self.x1 = x1_new
            self.x2 = x2_new
            self.x3 = x3_new
            self.x4 = x4_new
            self._rectify_x()
            self._updateMatrix()


            angular_velocities = np.array([[float(dtheta[0])], [float(dtheta[1])]]).reshape(2, 1)
            tau = self.M_theta @ angular_velocities + self.c_theta + self.g_theta
            self.tau_theta = tau
            tau1s.append(tau[0])
            tau2s.append(tau[1])
        return theta1s, theta2s, d_theta1s, d_theta2s, d_d_theta1s, d_d_theta2s, tau1s, tau2s

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

def getH(theta1, theta2, d_theta1, d_theta2,dd_theta1, dd_theta2, g):
    return np.array(
        [
            [dd_theta1,dd_theta1,math.cos(theta2)*(2*dd_theta1+dd_theta2)-math.sin(theta2)*(2*d_theta1*d_theta2+d_theta2**2),dd_theta1+dd_theta2,g*math.cos(theta1),g*math.cos(theta1),g*math.cos(theta1+theta2)],
            [0,0,math.cos(theta2)*dd_theta1+math.sin(theta2)*d_theta1**2,dd_theta1+dd_theta2,0,0,g*math.cos(theta1+theta2)]
        ]
    )

    return np.array(
        [
            [dd_theta1, math.cos(theta2) * (2 * dd_theta1 + dd_theta2) - math.sin(theta2) * (
                        2 * d_theta1 * d_theta2 + d_theta2 ** 2), dd_theta1 + dd_theta2, g * math.cos(theta1),
            g * math.cos(theta1 + theta2)],
            [0, math.cos(theta2) * dd_theta1 + math.sin(theta2) * d_theta1 ** 2, dd_theta1 + dd_theta2, 0,
             g * math.cos(theta1 + theta2)]
        ]
    )

def ls(H, y):
    return (H.T@H)**(-1)@H.T@y

def calParam(w):
    l1 = float(w[0])/float(w[4])
    m1 = float(w[4])/l1
    m2 = float(w[5])/l1
    l2 = float(w[6])/m2
    # l2 = float(w[2])/float(w[4])
    # m2 = float(w[4])/l2
    # l1 = float(w[1])/m2/l2
    # m1 = float(w[3])/l1 - m2
    return m1, m2, l1, l2


if __name__ == '__main__':
    steps = 2000
    T = 0.001

    d1 = Droping(T = T, theta1 = 0, theta2 = np.pi/2)
    x1s1,x2s1, x3s1, x4s1 = d1.forward(steps)

    time = np.arange(0, steps*T, T)

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

    # 将x3作为x坐标，x4作为y坐标，绘制轨迹
    plt.figure(figsize=(8,8))
    plt.plot(x3s1, x4s1)
    plt.title('trajectory')
    plt.show()

    d2 = Droping(T = T, theta1 = 0, theta2 = -np.pi/2)
    x1s2, x2s2, x3s2, x4s2 = d2.forward(steps)


    threhold = 1950
    time = time[threhold:]
    x1s2 = x1s2[threhold:]
    x2s2 = x2s2[threhold:]
    x3s2 = x3s2[threhold:]
    x4s2 = x4s2[threhold:]
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

    d3 = Droping(T = T, theta1 = 0, theta2 = np.pi/2)
    theta1s, theta2s, d_theta1s, d_theta2s, d_d_theta1s, d_d_theta2s, tau1s, tau2s = d3.forward_parameters(steps, T)
    # H = np.array([[theta1s], [theta2s], [d_theta1s], [d_theta2s]]).T
    print('shape of theta1s')
    print(np.array(theta1s).shape)
    target_point_start = 15

    H = None
    tau = None
    for i in range(15):
        target_point = 20*i + target_point_start
        Hi = getH(float(theta1s[target_point]), float(theta2s[target_point]),
                  float(d_theta1s[target_point]), float(d_theta2s[target_point]),
                  float(d_d_theta1s[target_point]), float(d_d_theta2s[target_point]), 9.81)
        taui = np.array([[float(tau1s[target_point])], [float(tau2s[target_point])]])
        if H is None:
            H = Hi
            tau = taui
        else:
            H = np.vstack((H, Hi))
            tau = np.vstack((tau, taui))



    print('shape of h:', H.shape)
    print('H*H.T:')
    print(H.T@H)
    print('(H*H^T)^-1: ', (H.T@H)**(-1))
    print('H: ',H)
    print('tau shape: ', tau.shape)
    w = ls(H,-tau)
    print('w: ', w)
    m1, m2, l1, l2 = calParam(w)

    print('m1 predict: ')
    print(m1)
    print('m2 predict: ')
    print(m2)
    print('l1 predict: ')
    print(l1)
    print('l2 predict: ')
    print(l2)

    # predict x1
    # ls = LeastSquare()
    # t_in = np.array(time)[:int(steps/2)]
    # x_in = np.array(x1s1)[:int(steps/2)]
    # y = np.array(x_in)
    # H = np.array([ls.transformH(x) for x in t_in])
    # theta, error = ls.fit(y=y, H=H)
    #
    # H_new = np.array([ls.transformH(x) for x in time])
    # x_predict = ls.predict(H_new)
    #
    # plt.figure(figsize=(8,8))
    # plt.plot(time, x_predict, label='predict')
    # plt.plot(time, x1s1, label='real')
    # plt.legend()
    # plt.show()


