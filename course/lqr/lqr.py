import numpy as np
import scipy.linalg as la
import control as control
import matplotlib.pyplot as plt

def getSlideKN(A, B, Q, R, N, nx, nu):
    P = np.zeros((nx,nx))
    K = np.zeros((nu,nx))
    for j in range(N):
        K = la.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
    return K

def getKN(A, B, Q, R, N, nx, nu):
    '''
    计算KN反馈增益矩阵。
    :param A: 状态空间矩阵A
    :param B: 状态输入矩阵B
    :param Q: 状态空间权重矩阵Q
    :param R: 状态输入权重矩阵R
    :param N: 迭代次数
    :param nx: 状态空间维度
    :param nu: 输入空间维度
    :return: 反馈增益矩阵KN
    '''
    P = np.zeros((nx,nx,N))
    K = np.zeros((nu,nx,N))
    for j in range(N):
        K[:,:,j] = la.inv(R + B.T @ P[:,:,j] @ B) @ B.T @ P[:,:,j] @ A
        P[:,:,j+1] = Q + A.T @ P[:,:,j] @ A - A.T @ P[:,:,j] @ B @ K[:,:,j]
    return K[:,:,N-1]

def lqrfunc(A, B, nx, nu, Q, R, Nr):
    P = np.zeros((nx,nx,Nr))
    for j in range(Nr-1):
        P[:,:,j+1] = Q + A.T @P[:,:,j] @ A - A.T @P[:,:,j] @ B @ la.inv(R + B.T @ P[:, :, j] @ B) @ B.T @ P[:, :, j] @ A
    print('manual solution:')
    print(P[:,:,Nr-1])

    Pstar = P[:,:,Nr-1]
    Kstar = la.inv(R + B.T @ Pstar @ B) @ B.T @ Pstar @ A

    print('use lib to solve:')
    R1, R2, R3 = control.dlqr(A,B,Q,R)
    print(R2)

    return Pstar, Kstar

if __name__ == '__main__':
    A = np.mat('1.95,-0.025,-1.6;16,1.1,-3.2;0.425,0.1875,0.3')
    B=np.mat('0 1 0;1 1 1').T
    nx = 3
    nu = 2
    Q = np.eye(nx)
    R = np.eye(nu)

    Nr = 20
    P = np.zeros((nx,nx,Nr))

    # for j in range(Nr-1):
    #     P[:,:,j+1] = Q + A.T @P[:,:,j] @ A - A.T @P[:,:,j] @ B @ la.inv(R + B.T @ P[:, :, j] @ B) @ B.T @ P[:, :, j] @ A
    # print('manual solution:')
    # print(P[:,:,Nr-1])
    #
    # Pstar = P[:,:,Nr-1]
    # Kstar = la.inv(R + B.T @ Pstar @ B) @ B.T @ Pstar @ A
    #
    # print('use lib to solve:')
    # R1, R2, R3 = control.dlqr(A,B,Q,R)
    # print(R2)
    Pstar, Kstar = lqrfunc(A, B, nx, nu, Q, R, Nr)

    # new
    N = 15
    x = np.mat(np.zeros((nx,N)))
    u = np.mat(np.zeros((nu,N)))
    norm_u = np.zeros(N)
    x[:,0] = np.mat('1;2;3')

    for k in np.arange(0,N-1):
        u[:,k] = -Kstar @ x[:,k]
        norm_u[k] = la.norm(u[:,k])
        x[:,k+1] = A @ x[:,k] + B @ u[:,k]

    # plot the result
    plt.figure()
    plt.subplot(221)
    plt.plot(np.arange(0,N),x[0,:].T)
    plt.title('x1')
    plt.subplot(222)
    plt.plot(np.arange(0,N),x[1,:].T)
    plt.title('x2')
    plt.subplot(223)
    plt.plot(np.arange(0,N),x[2,:].T)

    plt.title('x3')
    plt.subplot(224)
    plt.plot(np.arange(0,N),norm_u)
    plt.title('norm_u')
    plt.show()

    # change weighting
    # states
    Q = 2*np.mat('1 0 0;0 1 0;0 0 1')
    # larger controlling rate
    R = np.eye(nu)
    # Pnew = P[:,:,Nr-1]
    # Knew = la.inv(R + B.T @ Pnew @ B) @ B.T @ Pnew @ A
    Pnew, Knew = lqrfunc(A, B, nx, nu, Q, R, Nr)

    # N = 30
    xnew = np.mat(np.zeros((nx,N)))
    unew = np.mat(np.zeros((nu,N)))
    norm_unew = np.zeros(N)
    xnew[:,0] = np.mat('1;2;3')
    for k in np.arange(0,N-1):
        unew[:,k] = -Knew @ xnew[:,k]
        norm_unew[k] = la.norm(unew[:,k])
        xnew[:,k+1] = A @ xnew[:,k] + B @ unew[:,k]

    time = np.arange(N)
    plt.figure()
    plt.subplot(221)
    plt.plot(time,x[0,:].T)
    plt.plot(time,xnew[0,:].T)
    plt.title('x1')
    plt.subplot(222)
    plt.plot(time,x[1,:].T)
    plt.plot(time,xnew[1,:].T)
    plt.title('x2')
    plt.subplot(223)
    plt.plot(time,x[2,:].T)
    plt.plot(time,xnew[2,:].T)
    plt.title('x3')
    plt.subplot(224)
    plt.plot(time,norm_u)
    plt.plot(time,norm_unew)
    plt.title('norm_u')
    plt.show()

