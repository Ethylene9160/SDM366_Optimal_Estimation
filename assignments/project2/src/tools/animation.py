from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
import numpy as np
def plot_robot_arm_dynamics(num_frames, zs, thetas, L1, L2, step=50):
    for i in range(0, len(zs), step):
        plt.figure(figsize=(8, 6))
        joint_x = L1/2+zs[i]
        joint_y = 0
        theta = thetas[i]-float(np.pi/2)
        end_effector_x = zs[i]+L2*np.cos(theta)
        end_effector_y = L2*np.sin(theta)

        plt.plot([-L1/2+zs[i], joint_x], [0, joint_y], 'ro-')
        plt.plot([zs[i], end_effector_x], [0, end_effector_y], 'bo-')
        plt.plot(end_effector_x, end_effector_y, 'go')
        plt.xlim([-L1 - L2 , L1 + L2 ])
        plt.ylim([-L1 - L2 , L1 + L2 ])
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.title(f'2R Robotic Arm Movement at t={i * 0.001:.2f} seconds')
        plt.grid(True)
        plt.show()
        clear_output(wait=True)