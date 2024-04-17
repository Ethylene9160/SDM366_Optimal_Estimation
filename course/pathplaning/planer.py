import numpy as np

# path cost graph.
# costMatrix = np.array([
#     [0,4,-1,-1,-1,-1,6,-1],
#     [4,0,3,-1,-1,-1,-1,3],
#     [-1,3,0,6,-1,-1,-1,-1],
#     [-1,-1,6,0,2,-1,-1,5],
#     [-1,-1,-1,2,0,3,-1,4],
#     [-1,-1,-1,-1,3,0,4,-1],
#     [6,-1,-1,-1,-1,4,0,2],
#     [-1,3,-1,5,4,-1,2,0]
# ])

costMatrix = np.array([
    [0,     4,      np.inf, np.inf, np.inf, np.inf, 6,      np.inf  ],
    [4,     0,      3,      np.inf, np.inf, np.inf, np.inf, 3       ],
    [np.inf,3,      0,      6,      np.inf, np.inf, np.inf, np.inf  ],
    [np.inf,np.inf, 6,      0,      2,      np.inf, np.inf, 5       ],
    [np.inf,np.inf, np.inf, 2,      0,      3,      np.inf, 4       ],
    [np.inf,np.inf, np.inf, np.inf, 3,      0,      4,      np.inf  ],
    [6,     np.inf, np.inf, np.inf, np.inf, 4,      0,      2       ],
    [np.inf,3,      np.inf, 5,      4,      np.inf, 2,      0       ]
])

# 检验costMatrix是不是对称矩阵。必然，costMatrix[i,j]=costMatrix[j,i]，
# 因为在这里，两点之间的距离，不管是去还是返回，是相等的。单纯自检。
def checkSymmetric(matrix):
    return (matrix == matrix.T).all()

print(checkSymmetric(costMatrix))

def valueIter(Vfunc, costMatrix, xf):
    '''
    迭代计算代价函数。
    :param Vfunc: 上一步的cost
    :param costMatrix: 地图
    :param xf: 终点下标。例如张巍老师的课程中，终点是4号点，在计算机里xf=3，
    :return: 这一步的cost（各点到目标点的距离集合），和到a_xf最近的点的集合。
    '''
    n = costMatrix.shape[0]

    Vnex = np.inf*np.ones(n) # min values
    mustar = -1*np.ones(n) # target index
    # 遍历起点a_i
    for i in range(n):
        J = np.inf * np.ones(n)  # 某个点到目标点的距离
        # 如果起点就是目标点：
        if i == xf:
            Vnex[i] = Vfunc[i]
            mustar[i] = i
            continue
        # 否则，遍历目标点a_j
        for j in range(n):
            if i == j: # 起点和目标点相同，跳过，此时的J默认为最大值，这样可以避免选择最短路径时选到自己，进入死循环。
                continue
            J[j] = Vfunc[j]+costMatrix[j,i]

        Vnex[i] = np.min(J)# a_i到a_xf最近的时候的cost。即i到xf的最短距离。
        mustar[i] = np.argmin(J) # a_i到a_xf最近的时候，下一步的下标。
    # print("Vnex:\n", Vnex)
    # print("mustar:\n", mustar)
    return Vnex, mustar

# def Vfunc(x,xf):
#     if x == 3:
#         return 0
#     return np.inf

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
