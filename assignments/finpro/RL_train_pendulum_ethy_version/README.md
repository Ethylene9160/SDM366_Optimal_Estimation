# RL Windows Version

注意：这里面大部分东西都是重写的，和助教提供的结构有很大不同，如果有问题，就别用这个了。

## Windows Environment

python: 3.9.18

请确保您安装了：

```bash
pip install torch
pip install mujoco==3.1.3
pip install mujoco-py==2.1.2.14
pip install gymnasium==0.27.1
pip install omegaconf==2.3.0
pip install hydra-core==1.3.2
```

## 文件结构

`ethy_RL_pendulum_demo.py`：采用mujoco官方接口viewer进行可视化。

`ethy_RL_project_demo.py`：采用mujoco进行仿真，但使用openCV进行可视化。