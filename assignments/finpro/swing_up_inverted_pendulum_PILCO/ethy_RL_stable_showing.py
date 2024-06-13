from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import mujoco
import glfw
import cv2
import tools
import time


def init_glfw(width=600, height=480):
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

    mujoco.mjv_updateScene(mujoco_model, mujoco_data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
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
    xml_path = "inverted_swing_pendulum.xml"
    model_path = "stable_2024-06-09-22-42-04/temp_model_save_at_epoch_2000.pth"
    # model_path = "models/ethy_official_stable1.ethy"

    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if window:
        # obs_space_dims = model.nq
        obs_space_dims = 6
        action_space_dims = model.nu
        print('nq: ', model.nq)
        print('nu: ', model.nu)
        print('state: ', data.qpos.shape[0])
        agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=0.000, gamma=0.99)
        agent.load_model(model_path)
        total_num_episodes = int(10) # training epochs
        time_records = []
        for episode in range(total_num_episodes):
            # video_writer = cv2.VideoWriter(f'vedio_for_{episode}th.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,(600, 480))
            rewards = []
            log_probs = []
            states = []
            done = False
            data.time = 0
            print('The episode is:', episode)
            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            tools.random_state(data)
            while not done:
                step_start = time.time()
                state = tools.get_obs(data)
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)

                ######  Calculate the Reward #######
                reward = 1.0
                ###### End. The same as the official model ########

                # rewards.append(reward)
                # log_probs.append(log_prob)
                # states.append(state.copy())
                done = data.time > 5 or abs(data.qpos[1]) > 0.7    # Example condition to end episode
                # video_record(model, data, video_writer) # uncommit this to make vedio
                render(window, model, data) # commit this line to speed up the training
            # log_probs.append(log_prob)
            # agent.update(rewards, log_probs, states)
            time_records.append(data.time)
            print(f'lasted for {data.time:.2f} seconds')
            # video_writer.release()
        print(f'max lasted {np.max(np.array(time_records)):.2f}s')
        print(f'avg lasted {np.mean(np.array(time_records)):.2f}s')

        glfw.terminate()
        # agent.save_model(save_model_path)
        # tools.save_model(agent, save_model_path)
