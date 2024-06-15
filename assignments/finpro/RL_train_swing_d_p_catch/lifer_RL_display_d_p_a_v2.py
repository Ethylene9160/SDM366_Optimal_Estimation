from __future__ import annotations

import random
import time
import warnings

import cv2
import glfw
import mujoco
import numpy as np
from stable_baselines3 import DDPG

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


if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    model_path = "models/v2/temp_1718387515_steps_20000.pth"

    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if_random_seed = 1
    seed = random.randint(0, 100000) if if_random_seed else 69521
    print(f'The seed is: {seed}')

    if window:
        try:
            obs_space_dims = 10
            action_space_dims = model.nu

            # Load PPO models
            env = tools.CustomEnv(model, data)
            agent = DDPG.load(model_path, env=env)

            total_num_episodes = int(10)
            time_records = []

            for episode in range(total_num_episodes):
                done = False
                data.time = 0
                print('The episode is:', episode)
                mujoco.mj_resetData(model, data)
                tools.random_state(data, seed)

                while not done:
                    state = tools.get_obs_lifer(data)
                    state_theta_1 = np.arctan2(state[6], state[8])
                    state_theta_2 = np.arctan2(state[7], state[9])
                    print(f"theta: {state_theta_1 + state_theta_2}")

                    action, _states = agent.predict(state)
                    data.ctrl[0] = action[0]
                    mujoco.mj_step(model, data)

                    done = data.time > 120

                    render(window, model, data)
                    # time.sleep(1)

                time_records.append(data.time)
                print(f'lasted for {data.time:.2f} seconds')

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            glfw.destroy_window(window)

        glfw.terminate()
