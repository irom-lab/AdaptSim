"""
Test and visualize Acrobot Swing-Up environment.

"""
import argparse
from omegaconf import OmegaConf
from adaptsim.env.acrobot_env import AcrobotEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    # Initialize environment
    dt = 0.01
    max_t = 20
    env = AcrobotEnv(
        dt=dt,
        max_t=max_t,
        render=args.gui,
    )

    # Initialize task
    task = OmegaConf.create()
    task.spong_m = [0.5, 0.5]  # mass values used for Spong controller
    task.spong_b = [0.5, 0.5]  # joint damping values used for Spong controller
    task.true_m = [0.5, 0.5]  # mass values used for simulation
    task.true_b = [0.5, 0.5]  # joint damping values used for simulation
    task.gains = {}
    task.gains['k_q'] = 10  # does not affect much
    task.gains['k_e'] = 20  # matters, 5-30, diagonal with k_p
    task.gains['k_p'] = 50  # matters, 20-100
    task.gains['k_d'] = 5  # matters, 1-10

    # Reset environment
    obs = env.reset(task=task)

    # Run
    obs, reward, done, info = env.step(action=None)
