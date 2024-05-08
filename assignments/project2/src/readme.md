# 233
## Q1

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
\dot x_3=f_3(x,u)=
\frac {1}{m_c+m_p\sin^2(x_2)}[
	u+m_p\sin(x_2)(L\dot x_2^2-g\cos(x_2))
]
$$
$\dot x_4$
$$
\dot x_4=f_4(x,u)=
\frac{1}{L(m_c+m_p\sin^2(x_2))}[u\cos(x_2)+m_pL\dot x_2^2\sin(x_2)\cos(x_2)-(m_c+m_p)g\sin(x_2)]
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
\frac{\partial f_3}{\partial x_1}=0\\
\frac{\partial f_3}{\partial x_2}=
\frac{
	m_c+m_p\sin^2x_2-2m_p\sin x_2\cos x_2
}{
	(m_c+m_p\sin^2 x_2)^2
}\\\cdot
[
	u+m_p\sin x_2(L\dot x_2^2-g\cos x_2)+m_pg(\sin^2x_2-\cos^2x_2)
]
\\
\frac{\partial f_3}{\partial x_3}=0\\
\frac{\partial f_3}{\partial x_4}=0\\
\frac{\partial f_3}{\partial u}=\frac {1}{m_c+m_p\sin^2(x_2)}
$$
f4:
$$
\frac{\partial f_4}{\partial x_1}=0\\
\frac{\partial f_4}{\partial x_2}=
\frac 1L\cdot\frac{
	m_c+m_p\sin^2x_2-2m_p\sin x_2\cos x_2
}{
	(m_c+m_p\sin^2 x_2)^2
}\\\cdot
[
	u\cos x_2+m_pL\dot x_2^2\sin x_2\cos x_2-(m_c+m_p)g\sin x_2\\
	-\sin x_2+m_pL\dot x_2^2(\cos^2 x_2-\sin^2 x_2)-(m_c+m_p)g\cos x_2
]
\\
\frac{\partial f_4}{\partial x_3}=0\\
\frac{\partial f_4}{\partial x_4}=0\\
\frac{\partial f_4}{\partial u}=\frac 1L\cdot\frac {1}{m_c+m_p\sin^2(x_2)}
$$


