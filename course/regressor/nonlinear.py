import random

import torch
import mujoco
import numpy as np
from numpy.random import random

# import plt
import matplotlib.pyplot as plt
from scipy import optimize

# from scipy.optimize import optimize

# from sklearn.utils import optimize

# from scipy import optimize

# This is an implemtation of a simple question in ppt //todo page

# generate b and truePos
# b: 2x6 matrix, each column is a measurement



def _distance(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return np.sqrt(res)


def cost(theta, b, y):
    '''
    A function of cost. That is: J(theta) = sum(y - ||theta - b||^2)
    :param theta: the estimated position
    :param b: the measurement data
    :param y: the true position
    :return: the cost
    '''
    j = 0.0
    m = b.shape[1]
    for i in range(m):
        j += (y[i] - _distance(theta, b[:, i]))**2
    # print("j: ", j)
    return j

# def cost(theta, b, y):
#     m = b.shape[1]
#     res = 0
#     for i in range(m):
#         res += np.square(y[i] - np.linalg.norm(theta - b[:, i]))
#     print("res: ", res)
#     return res

# Deprecated.
def cost2(theta, b, y):
    m = b.shape[1] # Get the number of columns
    # np.linalg.norm calculate the L2 norm of a vector (计算向量的L2范数，也就是欧几里得距离)
    # ie. np.linalg.norm([1, 2, 3]) = sqrt(1^2 + 2^2 + 3^2)
    # theta[:, None] - b
    # i.e. theta = [1,2,3] => theta[:, None] = [[1], [2], [3]], or => [[1,1],[2,2],[3,3]], or .etc.
    # and '-b' means that,
    # thus, i.e. theta = [0,0], b = [[1,2,3], [4,5,6]]
    # => theta[:, None] - b =[[0,0,0], [0,0,0]] - [[1,2,3], [4,5,6]] = [[-1,-2,-3], [-4,-5,-6]]
    res = np.sum(np.square(y - np.linalg.norm(theta[:, None] - b, axis=0)))
    print("res: ", res)
    return res




