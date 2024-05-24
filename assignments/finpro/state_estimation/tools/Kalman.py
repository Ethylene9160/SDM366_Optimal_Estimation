import numpy as np
class Kalman:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        """
        Initialize the Kalman Filter
        Args:
            F: State Transition matrix
            B: Control Input matrix
            H: Observation model matrix
            Q: Process Noise Covariance
            R: Observation Noise Covariance
            P: Error Covariance Matrix
            x0: Initial State
        """
        self.n = F.shape[1]

        self.F = F
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
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        Update the Kalman Filter from a new observation z
        Args:
            z: Observation
        Return:
            Updated State
        """
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x