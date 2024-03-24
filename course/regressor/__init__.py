import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from numpy.random import random

import nonlinear as nlr # nonlinear regressor

if __name__ == '__main__':
    # generate b and truePos
    # b: 2x6 matrix, each column is a measurement
    b = np.mat(([1, -2, -4, 5, -7, 8], [3, -2, 3, 6, 5.5, 10]))
    # truePos: 2x1 matrix, the true position
    truePos = np.transpose(np.mat([4, 0]))

    # plot the original data
    plt.figure()
    plt.subplot(211)
    plt.plot(b[0, :], b[1, :], 'ro')
    plt.plot(truePos[0], truePos[1], 'bo', label='True position')
    plt.title('Measurement data (Original)')
    plt.legend()

    # generate mesuarment data
    m = b.shape[1]

    # generate noise
    noise = 0.1 * random(m)
    y = np.zeros(m)

    for i in range(m):
        y[i] = np.linalg.norm(truePos - b[:, i], axis=0) + noise[i]

    # least squares to get the result.
    H = np.array(b)
    print('shape of H: ', H.shape)
    thetahat_LS = np.linalg.inv(H.dot(H.T)).dot(H).dot(y)


    print("Estimated position (Least Squares): ", thetahat_LS)

    # optimize the result
    theta0 = np.array([0, 0])
    res = minimize(nlr.cost, theta0, args=(b, y), method='BFGS')
    print("Estimated position (Optimize): ", res.x)
    # res = thetahat_LS
    # use theta_LS to calculate the result

    # plot the result
    plt.subplot(212)
    plt.plot(b[0, :], b[1, :], 'ro')
    plt.plot(truePos[0], truePos[1], 'bo', label='True position')
    # plt.plot(thetahat_LS[0], thetahat_LS[1], 'go')
    plt.title('Estimated position')
    # plt.plot(res.x[0], res.x[1], 'yo', label='Estimated position')
    plt.plot(thetahat_LS[0],thetahat_LS[1], 'yo', label='Estimated position')
    plt.legend()
    plt.show()