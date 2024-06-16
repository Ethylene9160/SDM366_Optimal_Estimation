from __future__ import annotations

import random
import warnings

import cv2
import glfw
import mujoco
import numpy as np
from stable_baselines3 import PPO

import tools

warnings.filterwarnings("ignore")


def init_glfw(width=900, height=480):
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window


def render(window, mujoco_model, mujoco_data):
    width, height = glfw.get_framebuffer_size(window)
    scene = mujoco.MjvScene(mujoco_model, maxgeom=1000)
    context = mujoco.MjrContext(mujoco_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_updateScene(mujoco_model, mujoco_data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(buffer, None, mujoco.MjrRect(0, 0, width, height), context)
    buffer = np.flipud(buffer)
    cv2.imshow("MuJoCo Simulation", buffer)
    cv2.waitKey(1)


def video_record(mujoco_model, mujoco_data, video_writer):
    width, height = glfw.get_framebuffer_size(window)
    scene = mujoco.MjvScene(mujoco_model, maxgeom=1000)
    context = mujoco.MjrContext(mujoco_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_updateScene(mujoco_model, mujoco_data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(buffer, None, mujoco.MjrRect(0, 0, width, height), context)
    buffer = np.flipud(buffer)

    if video_writer is not None:
        video_writer.write(buffer)


def is_stable(data):
    state = tools.get_obs_lifer(data)
    x, theta1, theta2, v, omega1, omega2, sin_theta1, sin_theta2, cos_theta1, cos_theta2 = state
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    # print(f"x: {x}, theta1: {theta1}, theta2: {theta2}, v: {v}, omega1: {omega1}, omega2: {omega2}")
    if abs(theta1) < 0.5 and \
            abs(theta2) < 0.5 and \
            abs(v) < 1.0 and \
            abs(omega1) < 3.0 and \
            abs(omega2) < 3.0:
        return True
    # return not is_unstable(data)
    return False


def is_unstable(data):
    _, _, z = data.site_xpos[0]
    return z < 0.5


if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    throw_path = "models/v1/temp_1718446981_steps_2000896.pth"
    catch_path = "models/ethy/ethy_official_double_stable.pth"

    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if window:
        try:
            obs_space_dims = 10
            action_space_dims = model.nu

            env = tools.CustomEnv(model, data)
            agent_throw = PPO.load(throw_path, env=env)

            agent_catch = tools.A2C.A2CAgent(obs_space_dims, action_space_dims)
            agent_catch.load_model(catch_path)

            total_num_episodes = int(9)

            for episode in range(total_num_episodes):
                if_random_seed = 1
                seed = random.randint(0, 100000) if if_random_seed else 86033
                print(f'The seed is: {seed}')

                done = False
                data.time = 0
                print('The episode is:', episode)

                mujoco.mj_resetData(model, data)
                tools.random_state(data, seed)

                isStable = False

                while not done:

                    if isStable:
                        if is_unstable(data):
                            isStable = False
                            print("Catch failed")
                    else:
                        if is_stable(data):
                            isStable = True
                            print("Catching...")

                    state = tools.get_obs_lifer(data) if not isStable else tools.get10obs(data)

                    action, _ = agent_throw.predict(state) if not isStable else agent_catch.sample_action(state)
                    data.ctrl[0] = action[0] if not isStable else action

                    mujoco.mj_step(model, data)

                    done = data.time > 300
                    render(window, model, data)

                    # time.sleep(1)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            glfw.destroy_window(window)

        glfw.terminate()
