# Give me a basic framework for a Kalman filter
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt


class Kalman:
    def __init__(self, A, B, C, D, Q, R, P, x):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x
        self.history = []

    def predict(self, u):
        # Predict state ahead
        self.x = self.A * self.x + self.B * u
        # Prediction error covariance P_{k|k-1}
        self.P = self.A * self.P * self.A.T + self.Q

    def update(self, y, u):
        # y = z - self.C * self.x
        # Compute Kalman gain
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ la.inv(S)
        # Update estimate with measurement y
        self.x = self.x + K @ (y-self.C@self.x-self.D@u)
        # Update the error covariance
        I = np.matrix(np.eye(self.P.shape[0]))
        self.P = (I - K @ self.C) @ self.P
        self.history.append(self.x)

    def getHistory(self):
        return np.array(self.history)

if __name__ == '__main__':
    nx = 3
    ny = 2
    N = 100

    x = np.zeros((nx,1))
    y = np.zeros((ny,1))
    u = np.zeros((nx,1))
    v = np.zeros((ny,1))

    xhat = np.mat(np.zeros((nx, N)))
    xpred = np.mat(np.zeros((nx, N)))

    Phat = np.zeros((nx,nx,N))
    Ppred = np.zeros((nx,nx,N))

    K = np.zeros((nx,nx,N))
    P = np.zeros((nx,nx,N))

    # Initialize
    xhat[:,0] = np.ones((nx,1))
    P[:,:,0] = np.eye(nx)

    A = np.array([
        [0,1,0],
        [0,0,1],
        [-1,-2,-3]
    ])
    B = np.array([
        [0],
        [0],
        [1]
    ])
    C = np.array([
        [1,0,0],
        [0,1,0]
    ])
    D = np.array([
        [0],
        [0]
    ])

    P = np.eye(nx)
    R = np.eye(ny)
    Q = np.eye(nx)

    kalman = Kalman(A, B, C, D, Q, R, P, x)
    for i in range(N):
        kalman.predict(u)
        kalman.update(v, u)
        xhat[:,i] = kalman.x
        Phat[:,:,i] = kalman.P
    history = kalman.getHistory()
    plt.figure()
    plt.plot(history[:,0])
    plt.plot(history[:,1])





