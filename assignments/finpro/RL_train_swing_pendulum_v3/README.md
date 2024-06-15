# RL Windows Version

注意：这里面大部分东西都是重写的，和助教提供的结构有很大不同，如果有问题，就别用这个了。

注意，为了确认代码能够运行，我鞭策ChatGPT生成了一些代码。这些代码不能实现project的功能，仅供结构的参考

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

`tools\__init__.py`：包含代理和训练类。

`ethy_official_model?.pth`：作者训练的样板模型。

## 使用方法

首先，请前往文件夹：`RL_train_pendulum_ethy_version`，这个文件夹中的文件经过了作者重构，能够适配windows。

然后，您可以运行`ethy_RL_pendulum_demo.py`或者`ethy_RL_project_demo.py`进行测试。

作者提供了三个瞎（）（）训练的模型供您进行测试。名字叫`ethy_official_model?.pth`。您可以根据您的需要来玩耍。
需要注意的是，这里的训练方法非常简单（GPT写的），所以效果不太行。请根据您的情况，在`tools\init.py`中重写代理和训练的函数。

## 其它

如果您遇到了mujoco-py的冲突，请卸载mujoco-py：

```bash
pip uninstall mujoco-py
```
