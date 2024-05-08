# Project 2

## Question 1

According to the description, the dynamics of the system are given by:
$$
\ddot{z} = \frac{1}{m_c + m_p \sin^2 \theta} \left[ u + m_p \sin \theta \left( L \dot{\theta}^2 + g \cos \theta \right) \right]
$$

$$
\dot{\theta} = \frac{1}{L(m_c + m_p \sin^2 \theta)} \left[ -u \cos(\theta) - m_p L \dot{\theta}^2 \cos \theta \sin \theta - (m_c + m_p) g \sin \theta \right]
$$

where $m_p = 1$, $m_c = 10$, $g = 9.81$, and $L = 0.5$​.

We define the state variable $x$ as:
$$
x = [z, \pi - \theta, \dot z, \dot \theta]^T
$$

Next, we analyze the first derivative of the state variable.

$\dot x_1$:
$$
\dot x_1=x_3
$$
$\dot x_2$:
$$
\dot x_2=-x_4
$$
$\dot x_3$:
$$
\dot x_3=f_3(x,u)&=
\frac {1}{m_c+m_p\sin^2(x_2)}[
	u+m_p\sin(x_2)(L\dot x_2^2-g\cos(x_2))
]\\&=
\frac {1}{m_c+m_p\sin^2(x_2)}[
	u+m_p\sin(x_2)(L x_4^2-g\cos(x_2))
]
$$
$\dot x_4$:
$$
\dot x_4=f_4(x,u)&=
\frac{1}{L(m_c+m_p\sin^2(x_2))}[u\cos(x_2)+m_pL\dot x_2^2\sin(x_2)\cos(x_2)-(m_c+m_p)g\sin(x_2)]\\&=
\frac{1}{L(m_c+m_p\sin^2(x_2))}[u\cos(x_2)+m_pLx_4^2\sin(x_2)\cos(x_2)-(m_c+m_p)g\sin(x_2)]
$$

Thus, the equation $\dot x = f(x,u)$ is:
$$
\dot{\mathbf{x}} = \begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2 \\
\dot{x}_3 \\
\dot{x}_4
\end{bmatrix} =
\begin{bmatrix}
x_3 \\
-x_4 \\
\frac {1}{m_c + m_p \sin^2(x_2)} \left[ u + m_p \sin(x_2) (L x_4^2 - g \cos(x_2)) \right] \\
\frac{1}{L(m_c + m_p \sin^2(x_2))} \left[ u \cos(x_2) + m_p L x_4^2 \sin(x_2) \cos(x_2) - (m_c + m_p) g \sin(x_2) \right]
\end{bmatrix}
$$


## Question 2

For ${\dot x}_1$:
$$
\frac{\partial f_1}{\partial x_1}=0\\
\frac{\partial f_1}{\partial x_2}=0\\
\frac{\partial f_1}{\partial x_3}=1\\
\frac{\partial f_1}{\partial x_4}=0\\
\frac{\partial f_1}{\partial u} = 0\\
$$

For ${\dot x}_2$:
$$
\frac{\partial f_2}{\partial x_1}=0\\
\frac{\partial f_2}{\partial x_2}=0\\
\frac{\partial f_2}{\partial x_3}=0\\
\frac{\partial f_2}{\partial x_4}=-1\\
\frac{\partial f_2}{\partial u}=0
$$
For ${\dot x}_3$:
$$
\frac{\partial f_3}{\partial x_1}&=0\\
\frac{\partial f_3}{\partial x_2}&=
\frac{
	m_c+m_p\sin^2x_2-2m_p\sin x_2\cos x_2
}{
	(m_c+m_p\sin^2 x_2)^2
}\\
\cdot &
[
	u+m_p\sin x_2(L x_4^2-g\cos x_2)+m_pg(\sin^2x_2-\cos^2x_2)
]
\\
\frac{\partial f_3}{\partial x_3}&=0\\
\frac{\partial f_3}{\partial x_4}&=\frac {2}{m_c+m_p\sin^2x_2}
(m_p\sin x_2L x_4)\\
\frac{\partial f_3}{\partial u}&=\frac {1}{m_c+m_p\sin^2(x_2)}
$$
For ${\dot x}_4$:
$$
\frac{\partial f_4}{\partial x_1}=0\\
\frac{\partial f_4}{\partial x_2}=
\frac 1L\cdot\frac{
	m_c+m_p\sin^2x_2-2m_p\sin x_2\cos x_2
}{
	(m_c+m_p\sin^2 x_2)^2
}\\\cdot
[
	u\cos x_2+m_pL x_4^2\sin x_2\cos x_2-(m_c+m_p)g\sin x_2\\
	-\sin x_2+m_pL x_4^2(\cos^2 x_2-\sin^2 x_2)-(m_c+m_p)g\cos x_2
]
\\
\frac{\partial f_4}{\partial x_3}=0\\
\frac{\partial f_4}{\partial x_4}=\frac{2}{L(m_c+m_p\sin^2x_2)}\cdot 
m_pLx_4\sin x_2 \cos x_2
\\
\frac{\partial f_4}{\partial u}=\frac 1L\cdot\frac {\cos x_2}{m_c+m_p\sin^2(x_2)}
$$
According to the formular, $m_p=1,m_c=10,g=9.81,L=0.5$, and $\pmb x=0,\pmb u=0$, we conclude that:
$$
\hat A = \begin{bmatrix}
\frac{\partial f_1}{x_1}~\frac{\partial f_1}{x_2}~\frac{\partial f_1}{x_3}~\frac{\partial f_1}{x_4}\\
\frac{\partial f_2}{x_1}~\frac{\partial f_2}{x_2}~\frac{\partial f_2}{x_3}~\frac{\partial f_2}{x_4}\\
\frac{\partial f_3}{x_1}~\frac{\partial f_3}{x_2}~\frac{\partial f_3}{x_3}~\frac{\partial f_3}{x_4}\\
\frac{\partial f_4}{x_1}~\frac{\partial f_4}{x_2}~\frac{\partial f_4}{x_3}~\frac{\partial f_4}{x_4}\\
\end{bmatrix} \  and \ \ 
\hat B=
\begin{bmatrix}
\frac{\partial f_1}{u}\\
\frac{\partial f_2}{u}\\
\frac{\partial f_3}{u}\\
\frac{\partial f_4}{u}
\end{bmatrix}
$$
That is, according to the question,
$$
\hat A = \begin{bmatrix}
0 & 0 & 1 & 0\\
0 & 0 & 0 & -1\\
0 & -0.981 & 0 & 0\\
0 & -21.582 & 0 & 0 
\end{bmatrix} \  and \ \ 
\hat B=\begin{bmatrix}
0\\0\\
0.1\\
0.2
\end{bmatrix}
$$

## Question 3

Assuming that the sampling time duration is $\Delta T.$ Now we have:
$$
\frac{x[k+1]-x[k]}{\Delta T}=\hat Ax[k]+\hat Bu[k]\\
x[k+1]=(\hat A\Delta T+I)x[k]+\hat B\Delta Tu[k]
$$
That is,
$$
\hat A_d=\begin{bmatrix}

1&0&\Delta T&0\\
0&1&0&-\Delta T\\
0&-0.981\Delta T&1&0\\
0&-21.582\Delta T&0&1 
\end{bmatrix}\\
\hat B=\begin{bmatrix}
0\\0\\
0.1\Delta T\\
0.2\Delta T
\end{bmatrix}
$$

