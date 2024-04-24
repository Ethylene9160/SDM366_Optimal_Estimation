import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
class Dropping:
    def __init__(self, T, m1=1, m2=1, l1=0.5, l2=0.5, theta1 = 0.0, theta2 = 0.0, g=9.81):
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

        self.M_theta = np.array([[self.l1 ** 2 * m1 + m2 * (self.l1 ** 2 + 2 * l1 * l2 * math.cos(self.x2) + l2 ** 2),
                                m2 * (l1 * l2 * math.cos(self.x2) + l2 ** 2)],
                               [m2 * (l1 * l2 * math.cos(self.x2) + l2 ** 2), l2 ** 2 * m2]])
        self.c_theta = np.array([[-m2 * l1 * l2 * math.sin(self.x2) * (2 * self.x3 * self.x4 + self.x4 ** 2)],
                               [m2 * l1 * l2 * self.x3 ** 2 * math.sin(self.x2)]])
        self.g_theta = np.array([[g * l1 * (m1 + m2) * math.cos(self.x1) + g * l2 * m2 * math.cos(self.x1 + self.x2)],
                               [g * l2 * m2 * math.cos(self.x1 + self.x2)]])

        # acceleration
        ddtheta = np.array([[-g/l1],[0]])
        self.tau_theta =  self.M_theta @ ddtheta + self.c_theta + self.g_theta
        print('tau: ', self.tau_theta)
        test_ddtheta = np.linalg.inv(self.M_theta) @ (self.tau_theta - self.c_theta - self.g_theta)
        print('test_ddtheta:', test_ddtheta)
        # self.tau_theta = np.array([[0], [0]])
        # self.tau_theta = np.array([[0], [0]])

    def forward(self, steps, T):
        fg = 0
        t1 = []
        t2 = []
        dt1 = []
        dt2 = []
        ddt1 = []
        ddt2 = []
        tau1 = []
        tau2 = []
        for i in range(steps):
            # fg += 1
            x1_new = self.x1 + self.x3 * T
            x2_new = self.x2 + self.x4 * T

            self.x1 = x1_new
            self.x2 = x2_new
            t1.append(x1_new)
            t2.append(x2_new)

            # update x3, x4
            ddt = np.linalg.inv(self.M_theta) @ (self.tau_theta - self.c_theta - self.g_theta)



            # update x3, x4
            self.x3 = (self.x3 + ddt[0][0] * T)
            self.x4 = (self.x4 + ddt[1][0] * T)
            dt1.append(self.x3)
            dt2.append(self.x4)
            ddt1.append(ddt[0][0])
            ddt2.append(ddt[1][0])

            # update matrix
            self.M_theta = np.array([[self.l1 ** 2 * self.m1 + self.m2 * (self.l1 ** 2 + 2 * self.l1 * self.l2 * math.cos(self.x2) + self.l2 ** 2),
                                self.m2 * (self.l1 * self.l2 * math.cos(self.x2) + self.l2 ** 2)],
                               [self.m2 * (self.l1 * self.l2 * math.cos(self.x2) + self.l2 ** 2),
                                self.l2 ** 2 * self.m2]]).reshape((2, 2))

            self.c_theta = np.array([[-self.m2 * self.l1 * self.l2 * math.sin(self.x2) * (2 * self.x3 * self.x4 + self.x4 ** 2)],
                               [self.m2 * self.l1 * self.l2 * self.x3 ** 2 * math.sin(self.x2)]]).reshape((2, 1))
            self.g_theta = np.array([[self.g * self.l1 * (self.m1 + self.m2) * math.cos(self.x1) + self.g * self.l2 * self.m2 * math.cos(self.x1 + self.x2)],
                               [self.g * self.l2 * self.m2 * math.cos(self.x1 + self.x2)]]).reshape((2, 1))



            ddt = np.linalg.inv(self.M_theta) @ (self.tau_theta - self.c_theta - self.g_theta)

            self.tau_theta = self.M_theta @ ddt + self.c_theta + self.g_theta
            tau1.append(self.tau_theta[0][0])
            tau2.append(self.tau_theta[1][0])
            if fg % 200  is 0:
                print('tau: ', self.tau_theta)
                print('ddt: ', ddt)
            fg += 1
        return t1, t2, dt1, dt2, ddt1, ddt2, tau1, tau2

def loss(y, t):
    return np.mean((y-t)**2)

def cost(theta, tau, H):
    # 计算模型预测
    predictions = H @ theta
    # 计算误差
    errors = predictions - tau
    # 返回误差的平方和，作为优化的目标函数
    return np.sum(errors**2)


def getH(theta1, theta2, d_theta1, d_theta2,dd_theta1, dd_theta2, g):
    return np.array(
        [
            [dd_theta1,dd_theta1,math.cos(theta2)*(2*dd_theta1+dd_theta2)-math.sin(theta2)*(2*d_theta1*d_theta2+d_theta2**2),dd_theta1+dd_theta2,g*math.cos(theta1),g*math.cos(theta1),g*math.cos(theta1+theta2)]
            ,
            [0,0,math.cos(theta2)*dd_theta1+math.sin(theta2)*d_theta1**2,dd_theta1+dd_theta2,0,0,g*math.cos(theta1+theta2)]
        ]
    )

    return np.array(
        [
            [dd_theta1, math.cos(theta2) * (2 * dd_theta1 + dd_theta2) - math.sin(theta2) * (
                        2 * d_theta1 * d_theta2 + d_theta2 ** 2), dd_theta1 + dd_theta2, g * math.cos(theta1),
            g * math.cos(theta1 + theta2)]
            ,
            [0, math.cos(theta2) * dd_theta1 + math.sin(theta2) * d_theta1 ** 2, dd_theta1 + dd_theta2, 0,
             g * math.cos(theta1 + theta2)]
        ]
    )

def ls(H, y):
    return np.linalg.pinv(H) @ y
    return np.linalg.inv(H.T@H)@H.T@y

def calParam(w):
    w = w.reshape(-1)
    l1 = (w[0])/(w[4])
    m1 = (w[4])/l1
    m2 = (w[5])/l1
    l2 = (w[6])/m2
    return m1, m2, l1, l2
    l2 = (w[2])/(w[4])
    m2 = (w[4])/l2
    l1 = (w[1])/m2/l2
    m1 = (w[3])/l1 - m2
    return m1, m2, l1, l2


if __name__ == '__main__':
    steps = 2000
    T = 0.001
    time = np.arange(0, steps*T, T)

    dropping = Dropping(T, theta1 = 0, theta2 = math.pi/2)
    t1, t2, dt1, dt2, ddt1, ddt2, tau1, tau2 = dropping.forward(steps, T)


    plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.plot(time, t1)
    plt.title('theta1 varies with time')
    plt.subplot(212)
    plt.plot(time, t2)
    plt.title('theta2 varies with time')
    plt.show()


    H = None
    tau = None
    start_point = 200
    for i in range(10):
        target = 30*i+start_point
        hi = getH(t1[target], t2[target], dt1[target], dt2[target], ddt1[target], ddt2[target], 9.81)
        # taui = np.array([[ddt1[start_point]], [ddt2[start_point]]])
        taui = np.array([[tau1[target]], [tau2[target]]])
        if H is None:
            H = hi
            tau = taui
        else:
            H = np.vstack((H, hi))
            tau = np.vstack((tau, taui))

    start = 100
    end = 110
    print('t1:', t1[start:end])
    print('t2:', t2[start:end])
    print('dt1:', dt1[start:end])
    print('dt2:', dt2[start:end])
    print('ddt1:', ddt1[start:end])
    print('ddt2:', ddt2[start:end])

    print('rank of which: ', np.linalg.matrix_rank(H.T@H))
    print('H^T*H:', H.T@H)

    # 岭回归的实现
    lambda_ = 1e-5  # 正则化参数
    I = np.eye(H.shape[1])  # 单位矩阵，维度与 H 的列数相同

    # 计算岭回归解
    w = np.linalg.inv(H.T @ H + lambda_ * I) @ H.T @ tau

    theta0 = np.ones(H.shape[1]).astype(np.float64)
    w = minimize(cost, theta0, args=(tau, H)).x # optimizer

    # w = ls(H, tau) # least square
    print('w:', w)
    print('loss: ', loss(H@w, tau))

    m1, m2, l1, l2 = calParam(w)
    print('m1:', m1)
    print('m2:', m2)
    print('l1:', l1)
    print('l2:', l2)