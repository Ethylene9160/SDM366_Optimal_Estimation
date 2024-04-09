import numpy as np

import planer as pl

# In face, we can apply a smooth slide window.

if __name__ == "__main__":
    N = 4 # Max steps
    n = 8 # size of the graph (cost matrix)
    xf = 3 # final place index
    V = np.inf*np.ones(N)
    V[xf] = 0 # final place cost is 0

    path = []
    lastStep = 0
    for i in range(N-1):
        V, steps = pl.valueIter(V, pl.costMatrix, xf)
        path.append(steps[lastStep])
        lastStep = steps[lastStep]

    print(path)
