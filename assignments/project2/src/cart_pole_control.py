import math
import time
# import cv2
import numpy as np
import mujoco.viewer

import tools.pendulum

PATH = 'mujoco_file/cart_pole.xml'

if __name__ == '__main__':
    # load the model from
    model = mujoco.MjModel.from_xml_path(PATH)
    data = mujoco.MjData(model)

    ############# MY CODE BEGIN #############
    delta_theta = 0.1
    x0 = np.array([[0],[delta_theta],[0],[0]])
    T=model.opt.timestep
    print('T:', T)

    # set the Q and R matrix. Feel free to config it!
    R=np.eye(1)
    Q=np.diag([5, 1, 5, 1])

    # init the pendulum
    pendulum = tools.pendulum.Pendulum(x=x0, Q=Q, R=R, T=T)
    ############ END MY CODE #############

    # get the joint ids
    cart_id = 0
    pole_id = 1

    print("CartSlider", cart_id)
    print("PolePin", pole_id)


    # create viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        start = time.time()
        data.qpos[pole_id] = np.pi - delta_theta  # set initial position (you can try using initial state = 0)

        while viewer.is_running():
            step_start = time.time()

            ################### MY CODE BEGIN #############
            # get the speed, acc, angular speed and angular acc to calculate K.
            cart_velocity = data.qvel[cart_id]
            cart_acceleration = data.qacc[cart_id]
            pole_angular_velocity = data.qvel[pole_id]
            pole_angular_acceleration = data.qacc[pole_id]

            x_hat = np.array([[data.qpos[cart_id]],[math.pi-data.qpos[pole_id]],[cart_velocity],[pole_angular_velocity]])
            # set u
            data.ctrl[0] = -(pendulum.K@(x_hat))[0][0]
            ################### END MY CODE ###############
            mujoco.mj_step(model, data)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # print(time_until_next_step)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
