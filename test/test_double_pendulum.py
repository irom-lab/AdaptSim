"""
Test and visualize (Non-Linearized) Double Pendulum Swing-Up environment.

"""
import argparse
from omegaconf import OmegaConf

from adaptsim.env.double_pendulum_env import DoublePendulumEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    # Initialize environment
    dt = 0.01
    max_t = 2
    env = DoublePendulumEnv(
        dt=dt,
        max_t=max_t,
        render=args.gui,
        torque_limit=2e2,
    )

    # Initialize task
    task = OmegaConf.create()
    task.lqr_m = [1.0, 1.0]  # mass values used for LQR
    task.lqr_b = [1.0, 1.0]  # joint damping values used for LQR
    task.true_m = [1.0, 1.0]  # mass values used for simulation
    task.true_b = [1, 1]  # joint damping values used for simulation

    # Reset environment - this also calculates the LQR action
    obs = env.reset(task=task)

    # Action is calculated with LQR using mass and damping values from the task when resetting the environment
    obs, reward, done, info = env.step(action=None)
    print('Reward: ', reward)
