"""
Test and visualize (Non-Linearized) Pendulum Swing-Up environment.

"""
import argparse
from omegaconf import OmegaConf

from adaptsim.env.pendulum_env import PendulumEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    # Initialize environment
    dt = 0.01
    max_t = 2
    env = PendulumEnv(
        dt=dt,
        max_t=max_t,
        render=args.gui,
        torque_limit=2e2,
    )

    # Initialize task
    task = OmegaConf.create()
    task.lqr_m = 1.0
    task.lqr_b = 1.0
    task.true_m = 1.0
    task.true_b = 1.0

    # Reset environment - this also calculates the LQR action
    obs = env.reset(task=task)

    # Action is calculated with LQR using mass and damping values from the task when resetting the environment
    obs, reward, done, info = env.step(action=None)
    print('Reward: ', reward)