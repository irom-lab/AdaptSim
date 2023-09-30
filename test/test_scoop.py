"""
Test and visualize the scooping action in Scooping environment.

"""
import argparse
import time
import numpy as np
import pickle

from adaptsim.env.scoop_env import ScoopEnv

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
    env = ScoopEnv(
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
        mid_frame_timestamp=[0.2, 0.3, 0.4, 0.5, 0.75],
        traj_type='absolute',
        reward_type='simple',
        max_pad_center_dist=0.05,
        pad_center_reward_coeff=0.2,
    )
    env.seed(42)

    # Run
    s1 = time.time()
    num_trial = 10
    for ind in range(num_trial):
        print(f'Resetting {ind}...')
        task = task_all[ind]
        task.obj_mu = 0.25
        task.obj_modulus = 4
        obs = env.reset(task=task)

        pitch = 7  # deg
        pd1 = 0
        s_x_tip = 0.0
        action = np.array([pitch, pd1, s_x_tip])
        action[0] *= np.pi / 180
        obs, reward, _, info = env.step(action)

    print(f'Average time per trial: {(time.time() - s1) / num_trial}')
