from collections import namedtuple
import numpy as np
import torch
import logging

from adaptsim.policy.policy_base import PolicyBase
from adaptsim.learner.dyn_fit import DynFit


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])


class PolicyValueDynFit(PolicyBase):

    def __init__(self, cfg, venv, verbose=True):
        super().__init__(cfg, venv)

        # Learner
        self.learner = DynFit(cfg.learner)
        self.learner.build_network(cfg.learner.arch, verbose=verbose)
        self.module_all = [self.learner]  # for saving models

        # Parameters - no need to reset
        self.max_sample_steps = cfg.max_sample_steps
        self.batch_size = cfg.batch_size
        self.num_update = cfg.num_update

    #== Reset policy/optimizer/memory
    def reset_policy(self, policy_path=None):
        if policy_path:
            self.learner.load_network(policy_path)
            logging.info('Loaded policy network from: {}'.format(policy_path))
        else:
            self.learner.build_network(
                self.cfg.learner.arch, build_optimizer=False, verbose=False
            )
            logging.info('Built new policy network!')

    def reset_optimizer(self, optimizer_state=None):
        if optimizer_state:
            self.learner.load_optimizer_state(optimizer_state)
            logging.info('Loaded policy optimizer!')
        else:
            self.learner.build_optimizer()
            logging.info('Built new policy optimizer!')

    #== Learn, evaluate, run steps
    def learn(self, states, actions, **kwargs):
        self.learner.update(states, actions, self.batch_size, self.num_update)

    def evaluate(self, tasks=None, num_episode=None, verbose=False):
        """Re-use policy"""
        if tasks is not None:
            self.reset_tasks(tasks, verbose)
        if num_episode is None:
            num_episode = self.num_task

        # Run each task exactly once!
        self.set_eval_mode()
        num_episode_run, episode_info = self.run_steps(
            num_episode=num_episode, run_in_seq=True, force_random=False
        )
        assert num_episode_run == num_episode
        eval_reward_cumulative_avg = self.eval_reward_cumulative_all / num_episode

        if verbose:
            print('Avg reward: ', eval_reward_cumulative_avg)
        return eval_reward_cumulative_avg, episode_info

    def evaluate_real(self, tasks=None, verbose=False):

        # Reset tasks
        self.reset_tasks(tasks, verbose)

        # Run
        self.set_eval_mode()
        for task_id in range(self.num_task):
            task = self.task_all[task_id]

            # Print task info for real experiment
            logging.info(
                f'Please run task {task}, {task_id+1} out of {self.num_task}...'
            )

            # Use first env
            s, _ = self.reset_env(env_ind=0, task_id=task_id)
            logging.info(f'State: {s}')

            # Select action
            with torch.no_grad():
                a = self.forward(s, force_random=False)
            logging.info(f'Normalized action: {a}')
            a = self.venv.unnormalize_action(a)

            # Print task info for real experiment
            logging.info(f'Please apply raw action {a}...')

            # Wait to continue
            while 1:
                try:
                    d = input('Enter ctn to continue...')
                    if d == 'ctn':
                        break
                except:
                    continue

    def collect_real_data(
        self, tasks, num_traj_per_task=1, force_random=True, **kwargs
    ):
        """Do not save data. No belief."""

        # Repeat tasks for num_traj_per_task
        num_task = len(tasks)
        num_episode = num_task * num_traj_per_task
        tasks = [item for item in tasks for _ in range(num_traj_per_task)]

        # Run all tasks in sequence, no perturbing policy
        self.reset_tasks(tasks)
        self.set_eval_mode()
        action_all = []
        for task_id, task in enumerate(tasks):

            # Print task info for real experiment
            logging.info(
                f'Please run task {task}, {task_id+1} out of {num_episode}...'
            )

            # Use first env
            s, _ = self.reset_env(env_ind=0, task_id=task_id)
            logging.info(f'State: {s}')

            # Select action
            with torch.no_grad():
                a = self.forward(s, force_random=force_random)
            logging.info(f'Normalized action: {a}')
            a = self.venv.unnormalize_action(a)
            action_all += [a]

            # Print task info for real experiment
            logging.info(f'Please apply raw action {a}...')

            # Wait to continue
            while 1:
                d = input('Enter ctn to continue...')
                if d == 'ctn':
                    break
        return np.vstack(action_all)

    def collect_data(
        self, tasks, num_traj_per_task=1, force_random=True, traj_type='final',
        **kwargs
    ):

        # Repeat tasks for num_traj_per_task
        num_task = len(tasks)
        num_episode = num_task * num_traj_per_task
        tasks = [item for item in tasks for _ in range(num_traj_per_task)]

        # Run all tasks in sequence, no perturbing policy
        self.reset_tasks(tasks)
        self.set_eval_mode()
        num_episode_run, info_episode = self.run_steps(
            num_episode=num_episode, run_in_seq=True, force_random=force_random
        )
        r_episodes = np.array([info['reward'] for info in info_episode]
                             ).reshape(num_task, num_traj_per_task)
        logging.info(f'Collected {num_episode_run} trajectories!')

        # Reshape state_all into tasks - filter out unstable ones
        if traj_type == 'push_final':
            state_all = [info['bottle_T_final'][:2] for info in info_episode]
        elif traj_type == 'push_full':
            state_all = [
                np.vstack(info['bottle_T_all'])[None, :, :2]
                for info in info_episode
            ]

            # Fill values if trajectory is not full due to error by repeating last value
            num_step_expected = max(traj.shape[1] for traj in state_all)
            for ind in range(len(state_all)):
                if state_all[ind].shape[1] < num_step_expected:
                    state_all[ind] = np.concatenate((
                        state_all[ind],
                        np.tile(
                            state_all[ind][:, -1:, :], (
                                1, num_step_expected - state_all[ind].shape[1],
                                1
                            )
                        )
                    ), axis=1)
        else:
            raise 'Unknown trajectory type when collecting data!'
        state_all = torch.from_numpy(np.vstack(state_all)).float().to(
            self.device
        )  # num_episode x max_traj_len x obs_dim

        # Collect action - raw
        action_all = [info['action'][None] for info in info_episode]
        action_all = torch.from_numpy(np.vstack(action_all)
                                     ).float().to(self.device)

        # Collect parameters - not used
        param_all = torch.from_numpy(
            np.vstack([info['parameter'] for info in info_episode])
        ).float().to(self.device).reshape((num_task, num_traj_per_task, -1))
        param_all = param_all[:, 0, :]  # take the first traj of each task

        return param_all, state_all, action_all, r_episodes

    def run_steps(
        self, num_step=None, num_episode=None, force_random=True,
        run_in_seq=False
    ):
        if num_step is not None:
            cnt_target = num_step
        elif num_episode is not None:
            cnt_target = num_episode
        else:
            raise "No steps or episodes provided for run_steps()!"

        if run_in_seq:
            task_ids_yet_run = list(np.arange(num_episode))

        # Run
        info_episode = []
        cnt = 0
        while cnt < cnt_target:

            # Reset
            if run_in_seq:
                new_ids = task_ids_yet_run[:self.n_envs]
                task_ids_yet_run = task_ids_yet_run[self.n_envs:]
                s, task_ids = self.reset_env_all(new_ids)
            else:
                s, task_ids = self.reset_env_all()

            # Select action
            with torch.no_grad():
                a_all = self.forward(s, force_random=force_random)

            # Apply action - update heading
            s_all, r_all, done_all, info_all = self.step(a_all)

            # Check all envs
            for env_ind, (s_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):

                # Store the transition in memory if training mode - do not save next state
                action = a_all[env_ind]
                if not self.eval_mode:
                    self.store_transition(
                        s[env_ind].unsqueeze(0), action, r, None, done, info
                    )

                # Increment step count for the env
                self.env_step_cnt[env_ind] += 1

                # Check reward
                if self.eval_mode:
                    self.eval_reward_cumulative[env_ind] += r.item()
                    self.eval_reward_best[env_ind] = max(
                        self.eval_reward_best[env_ind], r.item()
                    )

                    # logging.info('Goal: {}, Loc: {}, Reward: {}'.format(info['goal'], info['bottle_T_final'][:2], r))
                    # logging.info('Action: {}'.format(info['action']))

                    # Check done for particular env
                    if done or self.env_step_cnt[env_ind] > self.max_env_step:
                        info['reward'] = self.eval_reward_cumulative[env_ind]
                        self.eval_reward_cumulative_all += self.eval_reward_cumulative[
                            env_ind]
                        self.eval_reward_best_all += self.eval_reward_best[
                            env_ind]
                        self.eval_reward_cumulative[env_ind] = 0
                        self.eval_reward_best[env_ind] = 0

                        # Record info of the episode
                        info_episode += [info]

                        # Count for eval mode
                        cnt += 1

                        # Quit
                        if cnt == cnt_target:
                            return cnt, info_episode

            # Count for train mode
            if not self.eval_mode:
                cnt += self.n_envs

                # Update gamma, lr etc.
                for _ in range(self.n_envs):
                    self.learner.update_hyper_param()
                    self.eps_schduler.step()

        return cnt, info_episode

    def forward(self, state, force_random=False):
        if force_random:
            a_all = self.venv.sample_random_action()
        else:
            a_all = self.learner(state)
        return a_all

    # === Replay and update ===
    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        transitions, _ = self.memory.sample(batch_size)
        return Transition(*zip(*transitions))

    def store_transition(self, *args):
        self.memory.update(Transition(*args))

    def unpack_batch(self, batch):
        state = torch.cat(batch.s).to(self.device)
        reward = torch.FloatTensor(batch.r).to(self.device)
        if self.action_dim == 1:
            action = torch.LongTensor(batch.a).to(self.device)
        else:
            action = torch.from_numpy(np.vstack(batch.a)
                                     ).long().to(self.device)

        # Optional
        append = None
        return (state, action, reward, append)
