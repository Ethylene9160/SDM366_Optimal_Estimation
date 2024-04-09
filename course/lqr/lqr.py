import numpy as np
import scipy.linalg as lg
import control as control

if __name__ == '__main__':
    A = np.mat('1.95,-0.025,-1.6;16,1.1,-3.2;0.425,0.1875,0.3')
    B=np.mat('0 1 0;1 1 1').T
    nx = 3
    nu = 2
    Q = np.eye(nx)
    R = np.eye(nu)

    Nr = 20
    P = np.zeros((nx,nx,Nr))

    for j in range(Nr-1):
        P[:,:,j+1] = Q+A.T@P[:,:,j]@A-A.T@P[:,:,j]@B@lg.inv(R+B.T@P[:,:,j]@B)@B.T@P[:,:,j]@A
    print('manual solution:')
    print(P[:,:,Nr-1])

    print('use lib to solve:')
    R1, R2, R3 = control.dlqr(A,B,Q,R)
    print(R2)