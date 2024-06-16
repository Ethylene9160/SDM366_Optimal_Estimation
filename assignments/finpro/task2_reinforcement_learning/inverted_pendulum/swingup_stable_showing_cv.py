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

def get_action(agent, data, stable_state):
    if stable_state:
        obs = tools.get6obs(data)
    else:
        obs = tools.get_obs(data)
    return agent.sample_action(obs)[0]

def is_stable(data):
    '''
    If the state of the pendulum is:
    abs(x) < 0.55
    abs(theta) < 0.24
    abs(v) < 1.8
    abs(omega) < 1.8
    Then it will be stable.
    '''
    if (abs(data.qpos[1]) < 0.45 and
            abs(data.qvel[0]) < 2.5 and
            abs(data.qvel[1]) <2.5):
        return True
    return False

def is_unstable(data):
    if abs(data.qpos[1]) > 0.8:
        return True
    return False
if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    stable_model_path = 'models/ethy_official_stable.ethy'
    swing_model_path = "models/ethy_official_swing_up.ethy"
    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if window:
        swing_agent = tools.DDPGAgent(4, 1, l_bound = 2.0, device = 'cuda')
        swing_agent.load_model(swing_model_path)
        stable_agent = tools.A2CAgent(6, 1)
        stable_agent.load_model(stable_model_path)
        agent = swing_agent
        total_num_episodes = int(10)
        time_records = []
        for episode in range(total_num_episodes):
            done = False
            stable_state = False
            agent = swing_agent
            data.time = 0
            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            data.qpos[1] = -np.pi
            state = tools.get_obs(data)
            while not done:
                step_start = time.time()

                action = get_action(agent, data, stable_state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                state = tools.get_obs(data)

                if stable_state == False:
                    if is_stable(data):
                        stable_state = True
                        agent = stable_agent
                        print('stable')
                else:
                    if is_unstable(data):
                        stable_state = False
                        agent = swing_agent
                        print('swing')

                done = data.time > 6  # Example condition to end episode
                render(window, model, data) # commit this line to speed up the training

        glfw.terminate()
