# Robot Dynamics Simulation and Parameter Indentification

## Q3 

（1）



（2）

Apply the method we mentioned in (1) in descrete time domain, we succeffully applyed it in jupyter notebook.

（3）

The system seems to be non-linear. In order to apply least square method, we're going to rectify the system:
$$
\tau=M(\theta)\ddot \theta +c(\theta,\dot\theta)+g(\theta)\\
=Ha
$$
with:
$$
H = \begin{bmatrix}
\ddot\theta_1,~\cos(2\ddot\theta_1+\ddot\theta_2)-\sin\theta_2(2\dot\theta_1\dot\theta_2),\ddot \theta_1+\ddot \theta_2,g\cos(\theta_1),g\cos(\theta_1+\theta_2)\\
0,\cos\theta_2\cdot\ddot\theta_1+\sin\theta_2\cdot\dot\theta_1^2,\ddot \theta_1+\ddot \theta_2,0,g\cos(\theta_1+\theta_2)
\end{bmatrix}
$$
and
$$
a=\begin{bmatrix}
m_1L_1^2+m_2L_1^2\\
m_2L_1L_2\\
m_2L_2^2\\
(m_1+m_2)L_1\\
m_2L_2
\end{bmatrix}=
\begin{bmatrix}
a_1\\a_2\\a_3\\a_4\\a_5
\end{bmatrix}
$$
Where we know,
$$
L_2 = \frac{a_3}{a_5}\\
m_2=\frac{a_5}{L_2}\\
L_1=\frac{a_2}{m_2L_2}\\
m_1=\frac{a_4}{L_1}-m_2
$$
Given $\tau=[2,1]^T$ as constant outerier torque, using $m_1, m_2=1kg,~L_1=L_2=0.5m$ as simulating parameters, and apply the method to perform least square method, we get:

|       | True Value | Predict Value |
| :---: | :--------: | :-----------: |
| $m_1$ |    1.00    |     0.98      |
| $m_2$ |    1.00    |      1.0      |
| $L_1$ |    0.50    |     0.48      |
| $L_2$ |    0.50    |     0.48      |

