"""
Test and visualize Linearized Double Pendulum Swing-Up environment.

"""
import argparse
import numpy as np
from omegaconf import OmegaConf

from adaptsim.env.double_pendulum_linearized_env import DoublePendulumLinearizedEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    # Initialize environment
    dt = 0.  # continue
    max_t = 2.5
    env = DoublePendulumLinearizedEnv(
        dt=dt,
        max_t=max_t,
    )

    # Initialize task
    task = OmegaConf.create()
    task.init_x = [-np.pi, 0, 0, 0]  # -np.pi from upright
    task.Q_gain = 1
    task.num_step_eval = 100
    task.true_m = [1.0, 1.5]
    task.true_b = [1.5, 1.5]

    # Reset environment - this also calculates the LQR action
    obs = env.reset(task=task)

    # Action is calculated with LQR using mass and damping values from the task, thus optimal
    obs, reward, done, info = env.step(action=None)
    print('Reward: ', reward)
