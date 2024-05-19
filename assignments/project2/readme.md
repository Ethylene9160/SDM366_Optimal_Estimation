# 233
## Q1

Assuming that:
$$
x=[z,\pi-\theta,\dot z, \dot \theta]
$$


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
$\dot x_4$
$$
\dot x_4=f_4(x,u)&=
\frac{1}{L(m_c+m_p\sin^2(x_2))}[u\cos(x_2)+m_pL\dot x_2^2\sin(x_2)\cos(x_2)-(m_c+m_p)g\sin(x_2)]\\&=
\frac{1}{L(m_c+m_p\sin^2(x_2))}[u\cos(x_2)+m_pLx_4^2\sin(x_2)\cos(x_2)-(m_c+m_p)g\sin(x_2)]
$$

## Q2

$$
\frac{\partial f_1}{\partial x_1}=0\\
\frac{\partial f_1}{\partial x_2}=0\\
\frac{\partial f_1}{\partial x_3}=1\\
\frac{\partial f_1}{\partial x_4}=0\\
\frac{\partial f_1}{\partial u}=0
$$

f2
$$
\frac{\partial f_2}{\partial x_1}=0\\
\frac{\partial f_2}{\partial x_2}=0\\
\frac{\partial f_2}{\partial x_3}=0\\
\frac{\partial f_2}{\partial x_4}=-1\\
\frac{\partial f_2}{\partial u}=0
$$
f3 
$$
\frac{\partial f_3}{\partial x_1}&=0\\
%%%%%%% f3 %%%%%%%%
\frac{\partial f_3}{\partial x_2}&=
% first term
\frac{
	-2m_p\sin x_2\cos x_2
}{
	(m_c+m_p\sin^2x_2)^2
}\cdot\\
&[u+m_p\sin x_2(Lx_4^2-g\cos x_2)]\\+
% second term
&\frac{
	1
}{
	m_c+m_p\sin^2x_2
}\cdot\\
&[m_p\sin x_2(g\sin x_2)+m_p\cos x_2(Lx_4^2-g\cos x_2)]


%%%%%%%%%%%% f3 end %%%%%%%%%%%%%%
\\
\frac{\partial f_3}{\partial x_3}&=0\\
\frac{\partial f_3}{\partial x_4}&=\frac {2}{m_c+m_p\sin^2x_2}
(m_p\sin x_2L x_4)\\
\frac{\partial f_3}{\partial u}&=\frac {1}{m_c+m_p\sin^2(x_2)}
$$
f4: 
$$
\frac{\partial f_4}{\partial x_1}&=0\\


%%%%%%%%%%%%%%%% x2 %%%%%%%%%%%%%

\frac{\partial f_4}{\partial x_2}=

% first term
&\frac{
	-2m_p\sin x_2\cos x_2
}{
	L(m_c+m_p\sin^2x_2)^2
}\cdot\\
&[u\cos x_2+m_pLx_4^2\sin x_2\cos x_2-(m_c+m_p)g\sin x_2]\\&+

% second term
\frac{1}{
	L(m_c+m_p\sin^2x_2)
}\cdot\\
&[-u\sin x_2+m_pLx_4^2(\cos^2x_2-\sin^2x_2)-(m_c+m_p)g\cos x_2]




%%%%%%%%%%%%%%%%%%% end x2

\\
\frac{\partial f_4}{\partial x_3}&=0\\
\frac{\partial f_4}{\partial x_4}&=\frac{2}{L(m_c+m_p\sin^2x_2)}\cdot 
m_pLx_4\sin x_2 \cos x_2
\\
\frac{\partial f_4}{\partial u}&=\frac 1L\cdot\frac {1}{m_c+m_p\sin^2(x_2)}
$$
According to the formular, $m_p=1,m_c=10,g=9.81,L=0.5$, and $\pmb x=0,\pmb u=0$ we conclude that:
$$
\hat A = \begin{bmatrix}
\frac{\partial f_1}{x_1}~\frac{\partial f_1}{x_2}~\frac{\partial f_1}{x_3}~\frac{\partial f_1}{x_4}\\
\frac{\partial f_2}{x_1}~\frac{\partial f_2}{x_2}~\frac{\partial f_2}{x_3}~\frac{\partial f_2}{x_4}\\
\frac{\partial f_3}{x_1}~\frac{\partial f_3}{x_2}~\frac{\partial f_3}{x_3}~\frac{\partial f_3}{x_4}\\
\frac{\partial f_4}{x_1}~\frac{\partial f_4}{x_2}~\frac{\partial f_4}{x_3}~\frac{\partial f_4}{x_4}\\
\end{bmatrix}\\
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
\hat A =
\begin{bmatrix}
0&0&1&0\\
0&0&0&-1\\
0&-\frac{m_pg}{m_c}&0&0\\
0&-\frac{(m_c+m_p)g}{Lm_c}&0&0 
\end{bmatrix}
=
\begin{bmatrix}
0&0&1&0\\
0&0&0&-1\\
0&-0.981&0&0\\
0&-21.582&0&0 
\end{bmatrix}\\
\hat B=
\begin{bmatrix}
0\\0\\
\frac{1}{m_c}\\
\frac{1}{Lm_c}
\end{bmatrix}=
\begin{bmatrix}
0\\0\\
0.1\\
0.2
\end{bmatrix}
$$

## Q3

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
0&21.582\Delta T&0&1 
\end{bmatrix}\\
\hat B=\begin{bmatrix}
0\\0\\
0.1\Delta T\\
0.2\Delta T
\end{bmatrix}
$$

## Q4

todo

## Q5

Firstly, we set the init condition as:
$$
x_0=[0,0.1,0,0]^T
$$
And we get 50 iteration to cauculate the feedback gain matrix $K$, using LQR method.

### Condition 1

We firstly, set the Q and R to be equal power i.e.
$$
Q=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}
~~~~R=\begin{bmatrix}
1
\end{bmatrix}
$$
And get the results.



### Condition 2

Secondly, we focus more on state space than input space,
$$
Q=\begin{bmatrix}
3&0&0&0\\
0&3&0&0\\
0&0&3&0\\
0&0&0&3
\end{bmatrix}
~~~~R=\begin{bmatrix}
1
\end{bmatrix}
$$
And get the results.

### Condition 3

Thirdly, we focus more on input state than state space,
$$
Q=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}
~~~~R=\begin{bmatrix}
3
\end{bmatrix}
$$
And get the results.

