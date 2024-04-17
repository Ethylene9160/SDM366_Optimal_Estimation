# Lecture Codes - 高仿张巍的代码dddd

课程代码的高仿版本。

3.？？（）记不清楚了：高仿了张巍课程中的rnn代码。`sdm_rnn`

3.24：高仿了张巍课程中regressor的代码，实现了一个简单的线性回归模型。`regressor`

4.9：高仿了张巍课程中，path-planning和lqr的代码。`pathplaining` and `lqr`

# RNN

# Regressor

# Path Planning

## Optimal Control Problem

<p align='center'><font size=5><b>Notation</b></font></p>

|    abbr.     |                        |                   备注                   |
| :----------: | :--------------------: | :--------------------------------------: |
| $l(x_k,u_k)$ | running cost function  |              每一步的开销。              |
|   $g(x_N)$   | ternimal cost function | 终端代价函数，即结束的时候，对应的开销。 |
| $J_N(x_0,u)$ |     N-horizon cost     |    目标函数。我们需要将它进行最小化。    |
|     $x$      |                        |                                          |
|     $u$      |                        |                                          |

在张巍的课堂中，我们学习了：
$$
J_N(x_0,u)=g(x_N)+\sum_{k=0}^{N-1}l(x_k,u_k)
$$

我们要做的是，最小化我们的**目标函数**。

在张巍老师的课堂上，其提出了这样几个问题：



现在以Path-planning问题，作为一个例子，来引入对这个问题的解答。

![image-path](readme.assets/path)

现在，我希望，从$a_1$走到$a_4$的最短路径。

最开始的状态，我们只知道相邻两点间的距离，而不相邻两点间的最短距离我们暂时不知道。现在我们假设Running cost用来表示：$l(z,u)$代表着z到u点的最短距离。如果没有遍历到，默认设置为$\infin$。

最终状态，我们假设，到了我们的目标点的时候，直接代价设为0，否则成本为$\infin$。即：
$$
g(z)=\begin{cases}
0,~if~z=a_4\\
\infin,~else
\end{cases}
$$
因而，我们的目的变成了，求取$J_N(x_0,u)$的最小值的时候，对应的各参数值。

## Dynamic Programming (动态规划)

事实上，如果将我们的函数进行进行“动态步骤移动”，具体到这道题目，假如每次我们都更新一轮到$a_4$点每个点最小距离（每次只走一步，这里的一步是指基于已经知道到$a_4$距离最近的点，更新其相邻点到$a_4$的距离。例如，第一步更新了$a_3~a_5~a_8$到$a_4$的距离，第二步更新$a_2$到$a_3$，$a_6$到$a_5$，$a_2~a_7$到$a_8$的距离，从而可以更新$a_2~a_6~a_7$到$a_4$的最短距离，将这样的距离保存下来），这样的思想叫做动态规划。

例如：

| Step | $a_1$    | $a_2$    | $a_3$ | $a_4$ |
| :--: | -------- | -------- | ----- | ----- |
|  1   | $\infin$ | $\infin$ | $5$   | $0$   |
|  2   | $\infin$ | $8$      | $5$   | $0$   |
|  3   | 12       | 8        | 5     | 0     |

那么，如何实现呢？

我们使用万能的双重for循环，遍历每两个点之间的距离，逐次更新即可。

首先，我们定义一个矩阵，用来表示两点之间的最短距离：

```python
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
```

其中，`costMatrix[i, j]`将表示$a_{i+1}$到$a_{j+1}$的距离。例如，$costMatrix[0, 1]$表示的是$a_1$到$a_2$​的最短距离。

这将被我们用来表示我们的running cost function, $l(x,u)=$​`costMatrix[x, u]`。

现在，让我们接着使用张巍老师的思路，以**最优**的思路来解决这个问题。
$$
J_N(x_0,u)=g(x_N)+\sum_{k=0}^{N-1}l(x_k,u_k)\\
x_{k+1}=f(x_k,u_k)\\
x_k\in X
$$
然后，





# LQR

<p align='center'><font size=5><b>Notation</b></font></p>

| abbr. |           含义           |   备注   |
| :---: | :----------------------: | :------: |
|  $L$  |                          |          |
|  $Q$  |    状态参数惩罚项系数    | 正定矩阵 |
|  $R$  |                          | 正定矩阵 |
|  $x$  |                          |          |
|  $u$  |                          |          |
| $P_j$ | 第j次的全状态参数惩罚项  | 正定矩阵 |
| $Q_f$ | 最终状态参数的惩罚项系数 |          |
| $V_j$ |       某一步的cost       |          |

考虑离散状态空间矩阵，假设我们设计的控制器有$u_{k}=-Kx_k$

$$
x_{k+1}=Ax_k+Bu_k=(A-BK)x_k\\
y_k=Cx_k+Du_k\tag{4-1}
$$

我们对过程进行“惩罚”，设定惩罚项。

$$
l(x,u)=x^TQx+u^TRu\tag{4-2}
$$

$Q,~R$需要是正定矩阵。这个式子告诉我们，只有$x,~u$趋近于零的时候，$l$才能取到最小值。此外，惩罚项的大小也对对应的被惩罚项衰减速率造成影响。例如，如果$Q>R$，那么x的衰减会比u更快，因为若每次希望降低相同的l，那么需要降低更多的x才能使l相对降低。

我们定义，经过N步之后：

$$
J_N(x_0,u)=x_N^TQ_fx_N+\sum_{k=0}^{N-1}[x_k^TQx_k+u_k^TRu_k]\tag{4-3}
$$

我们需要完成的，便是最小化代价函数，求出此时的参数$x,~u$。离散空间中，我将会使用$z$而不是$x$。所以接下来的推到中将出现的状态变量是$z$。我们假定我们每一步的cost function为：

$$
V_j(z)=z^TP_jz\tag{4-4}
$$

并定义迭代到下一步的cost为：

$$
V_{j+1}(z)=min\{l(z,u)+V_j(f(z,u))\}\\=min\{z^TQz+u^TRu+(Az+Bu)^TP_j(Az+Bu)\}\\=min\{u^T(R+B^TP_jB)u+2z^TA^TP_jBu+z^T(Q+A^TP_jA)z\}\tag{4-5}
$$

如果$V_{j+1}(z)$对u的偏导数为0，那么将会取得它关于u的极小值（证明略）。

$$
\frac{\partial h(u)}{\partial u}=2u^T(R+B^TP_jB)+2z^TA^TP_jB=0\\u^T=-z^TA^TP_jB(R+B^TP_jB)^{-1}\tag{4-6}
$$

由于R是对称矩阵（正定矩阵一定对称），$B^TP_jB$这个形式也是对称矩阵的标准形式，那么必然，$(R+B^TP_jB)$一定是对称矩阵，因此，$(R+B^TP_jB)^{-1}=(R+B^TP_jB)^{-T}$。同样，$P_j$也是对称矩阵。

$$
u_{j+1}=-(R+B^TP_jB)^{-1}B^TP_jAz=-K_{j+1}z\tag{4-6*}
$$

通过上式我们可以知道我们的控制器$u=-Kz$的反馈增益矩阵K应该如何设计了。这样设计的控制器可以保证每一步迭代都能使每一步的代价函数$V_j(z)$达到最小（根据(4-6)，这样设计的控制器可以让代价函数对输入的偏导数为0，从而达到它的极小值）。现在，让我们把$u=-Kz$代入（todo）：

$$
V_{j+1}(z)=min\{h(u^*)\}\\=(-K_jz)^T(R+B^TP_jB)(-K_jz)+2z^TA^TP_jB(-K_jz)+z^T(Q+A^TP_jA)z\\=z^T(Q+A^TP_jA-A^TP_jB(R+B^TP_jB)^{-1}B^TP_jA)z\tag{4-7}
$$

让我们对比（4-4）我们定义的代价函数，$V_{j+1}(z)=z^TP_{j+1}z$，我们震惊地发现，$P_j$和$P_{j+1}$之间存在迭代关系：

$$
P_{j+1}=Q+A^TP_jA-A^TP_jB(R+B^TP_jB)^{-1}B^TP_jA\tag{4-8}
$$

经过蛮长的迭代之后，我们可以得到最终的$P_N$。根据式(4-6*)，写出最终的输入和状态参数的表达关系式：

$$
u=-(R+B^TP_NB)^{-1}B^TP_NAz=-K_Nz\\K_N=(R+B^TP_NB)^{-1}B^TP_NA\tag{4-9}
$$

tips: (4-8)可以也被表示为：（不然代码太长了，物理意义））））



