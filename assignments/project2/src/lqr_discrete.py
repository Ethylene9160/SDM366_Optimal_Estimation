import numpy as np
import numpy.linalg as la

def getSlideKN(A, B, Q, R, N, nx, nu):
    P = np.zeros((nx,nx))
    K = np.zeros((nu,nx))
    for j in range(N):
        K = la.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
    K = la.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

def getDiscreteKN(A, B, Q, R, N, nx, nu):
    return getSlideKN(A, B, Q, R, N, nx, nu)
