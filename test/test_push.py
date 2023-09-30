"""
Test and visualize Tabletop Pushing environment.
Default to use the overlap env where a patch is added to the table.

"""
import argparse
import random
import time
from omegaconf import OmegaConf
from adaptsim.env.push_overlap_env import PushOverlapEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    # Initialize environment
    env = PushOverlapEnv(
        dt=0.01,
        render=args.gui,
        visualize_contact=False,
        hand_type='plate',
        diff_ik_filter_hz=200,
        contact_solver='tamsi',
        table_type='overlap',
        panda_joint_damping=1.0,
        flag_disable_rate_limiter=True,
        use_goal=True,
        max_dist_bottle_goal=0.2,
    )
    env.seed(42)

    # Run
    s1 = time.time()
    num_trial = 10
    for ind in range(num_trial):
        print('Resetting...')
        task = OmegaConf.create()
        task.obj_mu = 0.08
        task.obj_modulus = 4
        task.obj_com_x = 0
        task.obj_com_y = 0
        task.obj_com_z = -0.1
        task.overlap_mu = 0.2
        task.goal = [0.35, 0.10]
        obs = env.reset(task=task)

        # trajectory index
        vel_x = 0.5
        yaw = 0.
        action = [vel_x, yaw]
        s, reward, _, info = env.step(action)
    print('Average time per trial:', (time.time() - s1) / num_trial)
