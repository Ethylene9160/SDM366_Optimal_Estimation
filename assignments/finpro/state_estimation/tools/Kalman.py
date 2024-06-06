import numpy as np
import numpy.linalg as la
class Kalman:
    def __init__(self, A=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        """
        Initialize the Kalman Filter
        Args:
            A: State Transition matrix
            B: Control Input matrix
            H: Observation model matrix
            Q: Process Noise Covariance
            R: Observation Noise Covariance
            P: Error Covariance Matrix
            x0: Initial State
        """
        self.n = A.shape[1]

        self.A = A
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=0):
        """
        Predict the future state
        Args:
            u: Optional control vector
        Return:
            Updated State
        """
        # print('self.x - predict1: ', self.x)
        # print('self.A: ', self.A)
        # print('self.B: ', self.B)
        # print('u: ', u)
        self.x = self.A @ self.x + self.B @ u
        # print('self.x - predict2: ', self.x)
        self.P = (self.A @ self.P) @ (self.A.T) + self.Q
        return self.x

    def update(self, z):
        """
        Update the Kalman Filter from a new observation z
        Args:
            z: Observation
        Return:
            Updated State
        """
        # print('self.x - 1: ', self.x)
        y = z - self.H @ self.x
        S = self.R + self.H @ self.P @ self.H.T
        K = (self.P @ self.H.T) @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # print('self.x -2: ', self.x)
        I = np.eye(self.n)
        self.P = ((I - (K @ self.H)) @ self.P) @ (I - (K @ self.H)).T + (K @ self.R) @ K.T
        return self.x

class MKalman:
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