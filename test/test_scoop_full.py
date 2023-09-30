"""
Test and visualize the full scooping sequence in Scooping environment.

"""
import argparse
import numpy as np
import pickle

from adaptsim.env.scoop_full_env import ScoopFullEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    # Load tasks
    dataset = 'asset/veggie_task/v3_cylinder/1000.pkl'
    num_link = 1
    print("= Loading tasks from", dataset)
    with open(dataset, 'rb') as f:
        task_all = pickle.load(f)

    # Initialize environment
    env = ScoopFullEnv(
        dt=0.005,
        render=args.gui,
        visualize_contact=False,  # conflict with swapping geometry
        hand_type='panda',
        diff_ik_filter_hz=200,
        contact_solver='sap',
        panda_joint_damping=1.0,
        table_type='normal',
        flag_disable_rate_limiter=True,
        num_link=num_link,
        veggie_x=0.68,
        traj_type='absolute',
    )

    # Run
    task = task_all[0]
    obs = env.reset(task=task)
    action = np.array([8, 0.6, -0.1, -0.6])
    action[0] *= np.pi / 180
    _, reward, _, info = env.step(action)
    print('Reward: ', reward)