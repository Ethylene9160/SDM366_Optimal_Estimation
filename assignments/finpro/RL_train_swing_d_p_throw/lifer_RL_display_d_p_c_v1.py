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
    if abs(data.qpos[1]) < 0.2 and \
            abs(data.qpos[2]) < 0.2 and \
            abs(data.qvel[0]) < 0.5 and \
            abs(data.qvel[1]) < 1.0 and \
            abs(data.qvel[2]) < 1.0:
        return True
    # return not is_unstable(data)
    return False


def is_unstable(data):
    _, _, z = data.site_xpos[0]
    return z < 0.8


if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    throw_path = "models/v1/temp_1718442370_steps_501760.pth"
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

            total_num_episodes = int(3)

            for episode in range(total_num_episodes):
                if_random_seed = 1
                seed = random.randint(0, 100000) if if_random_seed else 94453
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

                    done = data.time > 30
                    render(window, model, data)

                    # time.sleep(1)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            glfw.destroy_window(window)

        glfw.terminate()
