import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sft

def f(x):
    return 3*(x**2)/(1+x**3)

def j(t,y):
    '''
    Calculate the error
    :param t: perdict value
    :param y: real value
    :return: square error
    '''
    return np.sum((t-y)**2)

def q1():
    '''
    Calculate the best fit of the function f(t) = 3*t^2/(1+t^3),
    This is for assignment 4, question 1-b
    :return: theta calculated by the 10 samples.
    '''
    ts = []
    ys = []
    H = []
    for i in range(10):
        ti = (1+i)*0.1
        yi = f(ti)
        ts.append(ti)
        ys.append(yi)
        H.append([1, ti**2, ti**3])
    # use numpy array
    H = np.array(H)
    ys = np.array(ys)
    ts = np.array(ts)

    # see result of H
    print('H:', H)

    # use least square method to find the best fit
    theta = np.linalg.inv(H.T@H)@H.T@ys

    # see result of a
    print('theta: ', theta)

    # calculate the error
    error = j(ts, H@theta)
    print('error:', error)
    return theta

def q1_advance(theta):
    '''
    Plot the best fit of the function f(t) = 3*t^2/(1+t^3)
    This is for the assignment 4, question 1-b, draw the plot
    :param theta: theta calculated by t = 0, 0.1, 0.2, ..., 1
    :return:
    '''

    ts = np.arange(0, 1.01, 0.01)
    y = f(ts)

    # construct H matrix
    H = np.column_stack((np.ones_like(ts), ts**2, ts**3))
    y_hat = np.dot(H, theta)

    plt.figure()
    plt.plot(ts, y, label='y')
    plt.plot(ts, y_hat, label='y_hat')
    plt.legend()
    plt.show()

def phi(k, N, n):
    return 1/N*np.exp(1j*2*np.pi*k*n/N)

def q2_theta(N, x):
    '''
    Calculate theta. Based on the manued output in the assignment,
    the elements in theta have been calculated.
    So we will not use theta = (H^* times H)^-1 times H^* times x to calculate the theta.
    :param N: N
    :param x: x
    :return: theta
    '''
    theta = []
    for k in range(N):
        theta.append(N*np.sum([x[n]*np.conj(phi(k, N, n)) for n in range(N)]))
    theta = np.array(theta)
    return theta

def q2():
    '''
    This is for the assignment 4, question 2
    :return: None
    '''
    N = 31
    n = np.arange(0, N, 1)
    print(n.shape)
    x = np.sin((1/5*np.pi)*n)
    # print(x.shape)

    plt.figure(figsize=(6,8))
    theta = q2_theta(N, x)
    # print(theta.shape)

    H = np.array([phi(k, N, n) for k in range(N)])
    # print(H.shape)

    x_hat = np.dot(H, theta)


    # print('======== My Results ========')
    # print('H: ', H)
    # print('predict y: ', x_hat)
    # print('loss:', j(abs(x), abs(x_hat)))
    #
    # print('======== Scipy Results ========')
    # x_hat_sft = sft.ifft(sft.fft(x))
    # print('predict y: ', x_hat_sft)
    # print('loss:', j(abs(x), abs(x_hat_sft)))

    x_hat_sft = sft.ifft(sft.fft(x))

    plt.subplot(311)
    plt.stem(n, x, label='x')
    plt.legend()
    plt.title('original x')
    plt.subplot(312)
    plt.stem(n, np.real(x_hat), label='x_hat')
    plt.legend()
    plt.title('x_hat')
    plt.subplot(313)
    plt.stem(n, np.real(x_hat_sft), label='x_hat_sft')
    plt.legend()
    plt.title('x_hat_sftusing scipy')
    plt.show()
    return H, x, x_hat

def q2_compare(H, x, x_hat):
    '''
    Compare the results of the assignment 4, question 2 with the scipy results
    :param x: x
    :param H: H
    :param y: y
    :return: None
    '''
    print('======== My Results ========')
    print('H: ', H)
    print('original y: ', x)
    print('predict y: ', x_hat)
    print('loss:', j(np.real(x), np.real(x_hat)))

    print('======== Scipy Results ========')
    x_hat_sft = sft.ifft(sft.fft(x))
    print('predict y: ', x_hat_sft)
    print('loss:', j(np.real(x), np.real(x_hat_sft)))



# See jupyter file.
if __name__ == '__main__':
    q1_advance(q1())
    q2()