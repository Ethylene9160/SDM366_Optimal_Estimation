import numpy as np

# path cost graph.
costMatrix = np.array([
    [0,4,-1,-1,-1,-1,6,-1],
    [4,0,3,-1,-1,-1,-1,3],
    [-1,3,0,6,-1,-1,-1,-1],
    [-1,-1,6,0,2,-1,-1,5],
    [-1,-1,-1,2,0,3,-1,4],
    [-1,-1,-1,-1,3,0,4,-1],
    [6,-1,-1,-1,-1,4,0,2],
    [-1,3,-1,5,4,-1,2,0]
])

# 检验costMatrix是不是对称矩阵
def checkSymmetric(matrix):
    return (matrix == matrix.T).all()

print(checkSymmetric(costMatrix))

def valueIter(Vfunc, costMatrix, xf):
    n = costMatrix.shape[0]
    J = -1*np.ones(n)
    Vnex = -1*np.ones(n) # min values
    mustar = -1*np.ones(n) # target index
    for xi in range(n):
        if xi == xf:
            Vnex[xi] = Vfunc[xi]
            mustar[xi] = xi
            continue
        for ui in range(n):
            # if xi == ui:
            #     continue
            xnext = ui
            if(costMatrix[xi,ui] == -1):
                # cost is infinite
                J[ui] = np.inf
            else:
                J[ui] = costMatrix[xi,ui] + Vfunc[xnext]
        # print("J:\n", J)
        Vnex[xi] = np.min(J)
        mustar[xi] = np.argmin(J)
    # print("Vnex:\n", Vnex)
    # print("mustar:\n", mustar)
    return Vnex, mustar

def Vfunc(x,xf):
    if x == 3:
        return 0
    return np.inf
if __name__ == '__main__':
    N=4# max step is 4
    n = 8 # size of the graph (costMatrix)
    x = -1*np.ones(N) # path
    x[0] = 0 # start point

    # cost function(init)
    g = np.inf*np.ones(n)
    g[3] = 0
    V = np.zeros((N,n))
    mu = np.zeros((N,n))
    V[0,:]=g
    # move 4 steps:
    for i in range(N-1):
        V[i+1,:],mu[i+1,:]=valueIter(V[i,:],costMatrix,3)

    for k in range(N-1):
        # get the next step. x[k] means the current position.
        x[k+1] = mu[N-k-1,x[k].astype(int)]
    print(x)
