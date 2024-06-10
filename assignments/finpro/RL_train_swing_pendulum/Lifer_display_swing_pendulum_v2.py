from __future__ import annotations

import random
import time
import warnings

import cv2
import glfw
import mujoco
import numpy as np

import tools

warnings.filterwarnings("ignore")


def init_glfw(width=1080, height=720):
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
    xml_path = "Lifer_inverted_swing_pendulum.xml"
    model_path = "models_v2/temp_1717904774_epoch_4224.pth"
    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if_random_seed = 1
    seed = random.randint(0, 100000) if if_random_seed else 39858
    print(f'The seed is: {seed}')

    if window:
        try:
            obs_space_dims = 6
            action_space_dims = model.nu
            agent = tools.DQNAgent(obs_space_dims, action_space_dims, lr=3e-4, gamma=0.99)
            agent.load_model(model_path)
            total_num_episodes = int(10)
            time_records = []
            for episode in range(total_num_episodes):

                # rewards = []
                # log_probs = []
                # states = []
                done = False
                data.time = 0
                print('The episode is:', episode)
                # 重置环境到初始状态
                mujoco.mj_resetData(model, data)
                tools.random_state(data, seed)
                while not done:
                    step_start = time.time()
                    state = tools.get_obs_lifer(data)
                    action = agent.sample_action(state)
                    data.ctrl[0] = action
                    mujoco.mj_step(model, data)

                    # rewards.append(reward)
                    # log_probs.append(log_prob)
                    # states.append(state.copy())

                    done = data.time > 450  # Example condition to end episode

                    # video_record(model, data, video_writer) # uncommitted this to record video
                    render(window, model, data)

                time_records.append(data.time)
                print(f'lasted for {data.time:.2f} seconds')
                # video_writer.release()

            print(f'max lasted {np.max(np.array(time_records)):.2f}s')
            print(f'avg lasted {np.mean(np.array(time_records)):.2f}s')

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            glfw.destroy_window(window)

        glfw.terminate()
