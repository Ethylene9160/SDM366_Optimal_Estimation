import numpy as np
import numpy.linalg as la
from . import Kalman
import mujoco

NOISE_PIMU_PRE = 0.01
NOISE_VIMU_PRE = 0.01
NOISE_PFOOT_PRE = 0.01
NOISE_PFOOD_REL = 0.01
NOISE_VFOOD_REL = 0.01
NOISE_ZFOOT_REL = 0.01

STATE_SIZE = 12
OBSERVE_SIZE = 14
class Human:
    def __init__(self, model, Pw):
        self.model = model
        self.T = model.opt.timestep

        # estimate state
        self.x = np.zeros((STATE_SIZE,1))
        self.x[0:3] = Pw

        # estimation state after process update
        self.xhat = np.zeros((STATE_SIZE,1))

        self.A = np.eye(STATE_SIZE)
        self.A[0:3, 3:6] = self.T * np.eye(3)

        self.B = np.zeros([STATE_SIZE, 3])
        self.B[3:6, 0:3] = self.T * np.eye(3)

        self.C = np.zeros((OBSERVE_SIZE,STATE_SIZE))
        self.C[0:3,0:3] = np.eye(3)
        self.C[3:6,0:3] = np.eye(3)
        self.C[0:6,6:12] = -np.eye(6)
        self.C[6:9,3:6] = np.eye(3)
        self.C[9:12,3:6] = np.eye(3)
        self.C[12,8] = 1
        self.C[13,11] = 1

        print('A: ', self.A)
        print('B: ', self.B)
        print('C: ', self.C)

        # estimation state covariance
        self.P = np.zeros((STATE_SIZE, STATE_SIZE))
        # estimation state covariance after process update
        self.Phat = np.zeros((STATE_SIZE, STATE_SIZE))

        # estimation state transition noise
        self.Q = np.zeros((STATE_SIZE, STATE_SIZE))

        self.y = np.zeros((OBSERVE_SIZE, 1))
        self.yhat = np.zeros((OBSERVE_SIZE, 1))
        self.error_y = np.zeros((OBSERVE_SIZE, 1))
        self.R = np.zeros((OBSERVE_SIZE, OBSERVE_SIZE))
        self.S = np.zeros((OBSERVE_SIZE, OBSERVE_SIZE))
        self.SC = np.zeros((OBSERVE_SIZE, STATE_SIZE))
    def _update_state(self):
        p_com = self.model.data.qpos[0:3]
        v_com = self.model.data.qvel[0:3]
        p_lf = np.zeros((3,1))
        p_rf = np.zeros((3,1))
        self.x = np.vstack([p_com, v_com, p_lf, p_rf])
        return self.x.copy()

    def _update_measurement(self):
        pass

    def run(self):
        noise_p = 0.01
        noise_v = 0.01
        noise_plf = 0.01
        noise_prf = 0.01
        Q = np.zeros([12,12], dtype=np.float32)
        Q[0:3, 0:3] = noise_p * np.eye(3)
        Q[3:6, 3:6] = noise_v * np.eye(3)
        Q[6:9, 6:9] = noise_plf * np.eye(3)
        Q[9:12, 9:12] = noise_prf * np.eye(3)

    def update(self, joint_angles, joint_velocities):
        pass

    def cal_reletive_p(self):
        pass

    def cal_w(self, Rb_imu, a_imu, w_imu):
        wb = Rb_imu @ w_imu


