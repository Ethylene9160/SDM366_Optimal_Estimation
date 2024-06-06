#   Copyright (c) 2024 CLEAR Lab
#  #
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#  #
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#  #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.


import numpy as np
import quaternion
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as Rot
import torch
import os
import time
import hydra
from omegaconf import DictConfig
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Set the directory for the simulation environment's source code
SOURCE_DIR = (
    ""  # TODO: change to your own project dir
)

urdf_path = "xiaotian/urdf/xiaotian.urdf"  # 替换为你的URDF文件路径
package_dirs = "xiaotian/meshes"
robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs, root_joint=pin.JointModelFreeFlyer())
# 初始化配置
# Flag to indicate whether to use the simulation environment
USE_SIM = False  # TODO: you should change to False to implement your own state estimator

timestep = 0.001


class MuJoCoSim:
    """Main class for setting up and running the MuJoCo simulation."""

    model: mujoco.MjModel
    data: mujoco.MjData
    policy: torch.nn.Module

    def __init__(self, cfg, policy_plan):
        """Class constructor to set up simulation configuration and control policy."""
        self.cfg = cfg  # Save environment configuration
        self.policy = policy_plan  # Save control policy

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            "xiaotian/urdf/xiaotian.xml"
        )
        self.model.opt.timestep = 0.001

        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        # Start the simulation viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Initialize the action array
        self.action = np.zeros(cfg.num_actions, dtype=np.double)

        # Control parameters
        self.Kp = np.array([40, 40, 40, 40, 40, 40], dtype=np.double)
        self.Kd = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.double)
        self.dof_pos_default = np.array([0, 0, 0, 0, 0, 0], dtype=np.double)
        self.tau_limit = 40.0 * np.ones(6, dtype=np.double)
        self.gait_index_ = 0.0

        # Simulation parameters
        self.decimation = 10
        self.action_scale = 0.25
        self.iter_ = 0

        # Set control commands
        self.commands = np.zeros((4,), dtype=np.double)
        self.commands[0] = 1.0  # x velocity
        self.commands[1] = 0.0  # y velocity
        self.commands[2] = 0.0  # yaw angular velocity
        self.commands[3] = 0.625  # base height

        # 自定义变量
        self.last_quaternion = np.quaternion(1, 0, 0, 0)
        self.lastBPL = np.zeros(3)
        self.lastBPR = np.zeros(3)
        self.x = np.array([0, 0, 0.625, 0, 0, 0, -0.0696, 0.095, 0, -0.0696, -0.095, 0]).T
        self.P = np.eye(12) * 0.01
        self.EKF_Q = np.eye(12)
        self.EKF_Q[:3, :3] = np.eye(3) * timestep * 0.02 / 20
        self.EKF_Q[3:6, 3:6] = np.eye(3) * timestep * 0.02 * 9.8 / 20
        self.EKF_Q[6:9, 6:9] = np.eye(3) * timestep * 0.02
        self.EKF_Q[9:12, 9:12] = np.eye(3) * timestep * 0.02

        self.EKF_R = np.zeros((14, 14))
        self.EKF_R[:6, :6] = np.eye(6) * 0.01
        self.EKF_R[6:12, 6:12] = np.eye(6) * 0.01
        self.EKF_R[12:14, 12:14] = np.eye(2) * 0.01
        self.W = np.zeros([3])

    def get_joint_state(self):
        """Retrieve the joint position and velocity states."""
        q_ = self.data.qpos.astype(np.double)
        dq_ = self.data.qvel.astype(np.double)
        return q_[7:], dq_[6:]

    def get_base_state(self):
        """Get the base state including linear velocity, angular velocity, and projected gravity."""
        quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)
        r = Rot.from_quat(quat)
        base_lin_vel = r.apply(self.estimate_base_lin_vel(), inverse=True).astype(  #
            np.double
        )
        base_ang_vel = self.data.qvel[3:6].astype(np.double)
        projected_gravity = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(
            np.double
        )
        return base_lin_vel, base_ang_vel, projected_gravity

    def get_contact(self):
        """Detect and return contact information."""
        num_contacts = self.data.ncon
        left_contact = 0
        right_contact = 0
        for i in range(num_contacts):
            contact = self.data.contact[i]
            # Determine contact based on geometry IDs
            if contact.geom2 == 5:
                left_contact = 1
            if contact.geom2 == 9:
                right_contact = 1
        return np.array([left_contact, right_contact])

    def compute_obs(self):  #
        """Calculate and return the observed states from the policy input."""
        obs_scales = self.cfg.observation.normalization
        dof_pos, dof_vel = self.get_joint_state()
        base_lin_vel, base_ang_vel, projected_gravity = self.get_base_state()  #
        CommandScaler = np.array(
            [
                obs_scales.lin_vel,
                obs_scales.lin_vel,
                obs_scales.ang_vel,
                obs_scales.base_height,
            ]
        )

        proprioception_obs = np.zeros([1, self.cfg.num_actor_obs], dtype=np.float32)
        proprioception_obs[0, :3] = projected_gravity * obs_scales.gravity  # 3
        proprioception_obs[0, 3:6] = base_ang_vel * obs_scales.ang_vel  # 3
        proprioception_obs[0, 6:12] = dof_pos * obs_scales.dof_pos  # 6
        proprioception_obs[0, 12:18] = dof_vel * obs_scales.dof_vel  # 6
        proprioception_obs[0, 18:22] = CommandScaler * self.commands  # 4
        proprioception_obs[0, 22:28] = self.action  # 6

        gait = np.array([2.0, 0.5, 0.5, 0.1])  # trot
        self.gait_index_ = self.gait_index_ + 0.02 * gait[0]
        if self.gait_index_ > 1.0:
            self.gait_index_ = 0.0
        gait_clock = np.zeros(2, dtype=np.double)
        gait_clock[0] = np.sin(self.gait_index_ * 2 * np.pi)
        gait_clock[1] = np.cos(self.gait_index_ * 2 * np.pi)

        proprioception_obs[0, 28:30] = gait_clock  # 2
        proprioception_obs[0, 30:34] = gait  # 4

        proprioception_obs[0, 34:37] = base_lin_vel

        return proprioception_obs

    def pd_control(self, target_q):
        """Use PD to find target joint torques"""
        dof_pos, dof_vel = self.get_joint_state()
        return (target_q + self.dof_pos_default - dof_pos) * self.Kp - self.Kd * dof_vel

    def run(self):
        """Main loop of simulation"""
        target_q = np.zeros(self.cfg.num_actions, dtype=np.double)
        while self.data.time < 1000.0 and self.viewer.is_running():
            step_start = time.time()
            proprioception_obs = self.compute_obs()  #
            if self.iter_ % self.decimation == 0:
                # proprioception_obs = self.compute_obs() #
                action = (
                    self.policy(torch.tensor(proprioception_obs))[0].detach().numpy()
                )
                self.action[:] = np.clip(
                    action, -self.cfg.clip.clip_actions, self.cfg.clip.clip_actions
                )
                target_q[:] = self.action_scale * self.action

            # Generate PD control
            tau = self.pd_control(target_q)  # Calc torques
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clip torques
            self.data.ctrl = tau

            mujoco.mj_step(self.model, self.data)

            if self.iter_ % 1 == 0:  # 50Hz 10
                self.viewer.sync()

            self.iter_ += 1
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def estimate_base_lin_vel(self):
        """Estimate the base linear velocity."""
        # Information you might need for doing state estimation
        contact_info = (
            self.get_contact()
        )  # [left_c, right_c] 1 is contact, 0 is off-ground
        # print(f'contact_info:{contact_info}')
        qpos_, qvel_ = (
            self.get_joint_state()
        )  # [left_abad, left_hip, left_knee, right_abad, right_hip, right_knee] 

        ang_vel, lin_acc = self.data.sensor("imu_gyro"), self.data.sensor("imu_acc")

        # 得到在Body坐标系下的左右脚的位置
        BPL, BPR, BVL, BVR = z2x_getFootPosition(qpos_[:3], qpos_[3:6], qvel_[:3], qvel_[3:6], contact_info)
        quat = quaternion.from_float_array(self.data.qpos.astype(np.double)[3:7])
        z, rotation_ob = z2x_Obserbation(BPL, BPR, BVL, BVR, ang_vel.data, quat, contact_info)
        # 加速度从自身坐标系变换到世界坐标系
        a = rotation_ob @ lin_acc.data + np.array([0, 0, -9.81])
        self.x, self.P = z2x_EKF(self.x, z, self.P, a, self.EKF_Q, self.EKF_R, contact_info)
        self.lastBPL, self.lastBPR = BPL, BPR
        # print('[debug] 机器人全局测量位置', self.x[:3])
        # print(['[debug] 机器人全局真实位置', self.data.qpos[:3]])
        if USE_SIM:
            return self.data.qvel[:3]
        else:
            return self.x[3:6]  # TODO: implement your codes to estimate base linear velocity


def z2x_EKF(x, z, P, u, Q, R, contact_info):
    # R 是测量协方差， Q是过程预测协方差，当处于摆动状态时，需要增大Q的方差，告诉模型现在过程不准
    if contact_info[0] == 0:
        Q[6:9, 6:9] = np.eye(3) * timestep * 1e10 * 0.02 / 20
        R[0:3, 0:3] = np.eye(3) * 0.01 * 1e10
        R[6:9, 6:9] = np.eye(3) * 0.01 * 1e10
        R[12, 12] = np.eye(1) * 0.01 * 1e10
        Q[9:12, 9:12] = np.eye(3) * timestep * 0.02 / 20
        R[3:6, 3:6] = np.eye(3) * 0.01
        R[9:12, 9:12] = np.eye(3) * 0.01
        R[13, 13] = np.eye(1) * 0.01
    if contact_info[1] == 0:
        Q[9:12, 9:12] = np.eye(3) * timestep * 1e10 * 0.02 / 20
        R[3:6, 3:6] = np.eye(3) * 0.01 * 1e10
        R[9:12, 9:12] = np.eye(3) * 0.01 * 1e10
        R[13, 13] = np.eye(1) * 0.01 * 1e10
        Q[6:9, 6:9] = np.eye(3) * timestep * 0.02 / 20
        R[0:3, 0:3] = np.eye(3) * 0.01
        R[6:9, 6:9] = np.eye(3) * 0.01
        R[12, 12] = np.eye(1) * 0.01
    R = R * 1e10

    A = np.eye(12)
    A[0:3, 3:6] = np.eye(3) * timestep
    B = np.zeros([12, 3])
    B[3:6, :] = np.eye(3) * timestep
    global z2x_x, z2x_v
    z2x_x = 0
    z2x_v = 0
    z2x_v += u[0] * timestep
    z2x_x += z2x_v * timestep
    x_hat = A @ x + B @ u
    P_hat = A @ P @ A.T + Q
    H = np.vstack((
        np.hstack((np.eye(3), np.zeros([3, 3]), -np.eye(3), np.zeros([3, 3]))),
        np.hstack((np.eye(3), np.zeros([3, 3]), np.zeros([3, 3]), -np.eye(3))),
        np.hstack((np.zeros([3, 3]), np.eye(3), np.zeros([3, 3]), np.zeros([3, 3]))),
        np.hstack((np.zeros([3, 3]), np.eye(3), np.zeros([3, 3]), np.zeros([3, 3]))),
        np.hstack((np.zeros([1, 8]), np.array([[1]]), np.zeros([1, 3]))),
        np.hstack((np.zeros([1, 11]), np.array([[1]])))
    ))
    K = P_hat @ H.T @ np.linalg.inv(H @ P_hat @ H.T + R)
    x = x_hat + K @ (z - H @ x_hat)
    P = (np.eye(12) - K @ H) @ P_hat
    return x, P


def z2x_Obserbation(BPL, BPR, BVL, BVR, W, quat, contact_info):
    matrix_q = quaternion.as_rotation_matrix(quat)
    z = np.concatenate((-matrix_q @ BPL,
                        -matrix_q @ BPR,  # error?
                        -matrix_q @ (np.cross(W, BPL) + BVL),
                        -matrix_q @ (np.cross(W, BPR) + BVR),
                        np.array([0.037062]),
                        np.array([0.037062])
                        ), axis=0)
    return z, matrix_q


def z2x_getFootPosition(leftP, rightP, leftV, rightV, contact):
    """
    得到在Body坐标系下的左右脚的位置和速度
    """

    qpos = np.zeros(robot.model.nq)
    qvel = np.zeros(robot.model.nv)

    qpos[:3] = leftP
    qpos[7:10] = rightP
    qvel[:3] = leftV
    qvel[7:10] = rightV

    # 计算正运动学和雅可比矩阵
    pin.forwardKinematics(robot.model, robot.data, qpos, qvel)
    pin.updateFramePlacements(robot.model, robot.data)

    foot_L_id = robot.model.getFrameId("foot_L")
    foot_R_id = robot.model.getFrameId("foot_R")

    foot_L_to_base = robot.data.oMf[foot_L_id]  # 从base_Link到foot_L的转换矩阵
    foot_R_to_base = robot.data.oMf[foot_R_id]  # 从base_Link到foot_R的转换矩阵

    # 计算Jacobian矩阵
    J_foot_L = pin.computeFrameJacobian(robot.model, robot.data, qpos, foot_L_id, pin.LOCAL)
    J_foot_R = pin.computeFrameJacobian(robot.model, robot.data, qpos, foot_R_id, pin.LOCAL)

    # 提取左右脚的速度
    v_foot_L = J_foot_L[:3, :] @ qvel  # 提取线速度部分
    v_foot_R = J_foot_R[:3, :] @ qvel  # 提取线速度部分

    foot_L_pos_body = foot_L_to_base.translation
    foot_L_pos_body = np.array([-foot_L_pos_body[1], -foot_L_pos_body[0], foot_L_pos_body[2]])
    foot_R_pos_body = foot_R_to_base.translation
    foot_R_pos_body = np.array([foot_R_pos_body[1], foot_R_pos_body[0], foot_R_pos_body[2]])
    v_foot_L_body = foot_L_to_base.rotation @ v_foot_L
    v_foot_L_body = np.array([v_foot_L_body[1], v_foot_L_body[0], v_foot_L_body[2]])
    v_foot_R_body = foot_R_to_base.rotation @ v_foot_R
    v_foot_R_body = np.array([v_foot_R_body[1], v_foot_R_body[0], v_foot_R_body[2]])
    # 输出结果是[3,]维度
    return foot_L_pos_body, foot_R_pos_body, v_foot_L_body, v_foot_R_body


@hydra.main(
    version_base=None,
    config_name="xiaotian_config",
    config_path="cfg",
)
def main(cfg: DictConfig) -> None:
    policy_plan_path = "policy/policy.pt"
    policy_plan = torch.jit.load(policy_plan_path)
    sim = MuJoCoSim(cfg.env, policy_plan)
    sim.run()


if __name__ == "__main__":
    main()
