import numpy as np
import mujoco.viewer
from scipy.linalg import solve_continuous_are
from control import place, lqr
import matplotlib.pyplot as plt
import time

# load the model from
model = mujoco.MjModel.from_xml_path("mujoco_file/test.xml")
data = mujoco.MjData(model)

# get the joint ids
cart_id = 0
pole_id = 1

print("CartSlider", cart_id)
print("PolePin", pole_id)

# system parameters
g = 9.81  # gravity
l = 0.5  # length of pole rod
m = 1.0  # mass of pole
M = 10.0  # mass of cart

# TODO: here is where you should code something
u = 0

# create viewer
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    start = time.time()
    data.qpos[pole_id] = np.pi - 0.5 # set initial position (you can try using initial state = 0)
    while viewer.is_running():
            step_start = time.time()
            data.ctrl[0] = u
            mujoco.mj_step(model, data)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        
