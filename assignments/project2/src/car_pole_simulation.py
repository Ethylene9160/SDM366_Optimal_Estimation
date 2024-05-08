import numpy as np
import numpy.linalg as la
import mujoco.viewer
import time

import tools.pendulum

PATH = 'mujoco_file/cart_pole.xml'

if __name__ == '__main__':

    ############# MY CODE BEGIN #############
    x0 = np.array([[0],[0.1],[0],[0]])
    T=0.002
    Q=3*np.eye(4)
    R=np.eye(1)
    print('shape of R: ', R)
    pendulum = tools.pendulum.Pendulum(x=x0, T = T)
    ############ END MY CODE #############

    # load the model from
    model = mujoco.MjModel.from_xml_path(PATH)
    data = mujoco.MjData(model)

    # get the joint ids
    cart_id = 0
    pole_id = 1

    print("CartSlider", cart_id)
    print("PolePin", pole_id)


    # create viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        start = time.time()
        data.qpos[pole_id] = np.pi - 0.1  # set initial position (you can try using initial state = 0)
        while viewer.is_running():
            step_start = time.time()

            ################### MY CODE BEGIN #############
            xi,ui=pendulum.step_in()
            data.qpos[pole_id] = float(np.pi)-xi[1][0]
            data.qpos[cart_id] = xi[0][0]
            ################### END MY CODE ###############
            mujoco.mj_step(model, data)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # print(time_until_next_step)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
