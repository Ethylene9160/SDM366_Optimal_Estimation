import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

if __name__ == '__main__':
    mu = np.array([5,5])
    sigma = np.array([[1.5,-0.5],[-0.5,1.5]])

    dist = multivariate_normal(mean=mu, cov=sigma)

    x, y = np.mgrid[0:10:.01, 0:10:.01]
    pos = np.dstack((x, y))
    Z = dist.pdf(pos)
    # draw 3d graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, Z, cmap='viridis')
    plt.show()


