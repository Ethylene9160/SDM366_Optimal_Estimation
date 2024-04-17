import numpy as np

import planer as pl

# In face, we can apply a smooth slide window.

if __name__ == "__main__":
    N = 4 # Max steps
    n = 8 # size of the graph (cost matrix)
    xf = 3 # final place index, we want to go to a_4, so the index is 3.

    # step cost function
    # V[xi] represents the minimum distance from a_xi to a_xf
    # i.e. V[0] represents the minimum distance from a_0 to a_3
    V0 = np.inf * np.ones(n)
    V0[xf] = 0 # a_xf to a_xf: distance is 0. else, are inf.

    path = []
    lastStep = 0
    steps = -1 * np.ones(n)
    x0 = -1*np.zeros(N)
    for i in range(N-1):
        V0, steps = pl.valueIter(V0, pl.costMatrix, xf)
        # print('steps: ',steps)
        # x0[i + 1] = steps[x0[i].astype(int)]
        # print('lastStep:', lastStep)
        # target = int(steps[lastStep])
        # path.append(target)
        # lastStep = target
    path.append(lastStep)
    for i in range(N-1):
        target = int(steps[lastStep])
        path.append(target)
        lastStep = target
    #
    #
    #
    print(path)

    # zw
    V = np.zeros((N, n))
    mu = np.zeros((N, n))
    V[0, :] = V0
    x = -1 * np.ones(N)  # path
    x[0] = 0  # start point
    for i in range(N-1):
        V[i+1,:],mu[i+1,:]=pl.valueIter(V[i,:],pl.costMatrix,3)
        print('mus: ',mu[i+1,:])
    for k in range(N-1):
        # get the next step. x[k] means the current position.
        x[k+1] = mu[N-1,x[k].astype(int)]
    print(x)