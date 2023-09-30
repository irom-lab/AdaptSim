import logging
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import numpy as np


class PolicyLinearized():

    def __init__(self, cfg, venv):
        """
        Policy for Linearized Double Pendulum. Train using policy gradient.
        """
        self.venv = venv

        # Basic
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.device = cfg.device

        # Params for vectorized envs
        self.n_envs = cfg.num_env
        self.max_steps_train = 1
        self.max_steps_eval = 1

        # Training
        self.lr = cfg.lr
        self.grad_clip = cfg.grad_clip
        # self.smoothing_coeff = cfg.smoothing_coeff
        self.action_dim = cfg.action_dim
        self.state_dim = cfg.state_dim
        self.epoch = cfg.epoch
        self.num_episode_per_epoch = cfg.num_episode_per_epoch
        self.Q_gain = cfg.Q_gain
        self.Q = np.diag([cfg.Q_gain for _ in range(cfg.state_dim)])
        self.R = np.diag([1.0 for _ in range(cfg.action_dim)])
        self.perturb_mag = cfg.perturb_mag
        self.num_step_eval_low = cfg.num_step_eval_low
        self.num_step_eval_high = cfg.num_step_eval_high

        # Load tasks
        if hasattr(cfg, 'dataset'):
            logging.info("= loading tasks from {}".format(cfg.dataset))
            with open(cfg.dataset, 'rb') as f:
                task_all = pickle.load(f)
            self.reset_tasks(task_all)
        else:
            logging.warning('= No dataset loaded!')

    @property
    def init_x(self):
        return [-np.pi, 0, 0, 0]

    def reset_policy(self, policy=None, sample_task=None):
        if policy is not None:
            self.K = deepcopy(policy)
        else:
            # self.K = self.rng.random(size=(self.action_dim, self.state_dim))
            # self.K = np.array(
            #     [[58.98510551, 19.49796874, 23.42134314,  8.80005279],
            #     [19.49796874, 19.98916802,  8.80005279,  5.82123757]]
            # )   # 1,1,1,1,dt=0

            #! Use true K from the mean of the distribution as initialization - for MoG, this is the mean of the mixture with highest weight - use a random environment to get K
            sample_task.init_x = self.init_x
            sample_task.Q_gain = self.Q_gain
            sample_task.true_m = [
                sample_task.param_mean[0], sample_task.param_mean[1]
            ]
            sample_task.true_b = [
                sample_task.param_mean[2], sample_task.param_mean[3]
            ]
            sample_task.num_step_eval = self.num_step_eval_high
            self.venv.reset_one(0, sample_task)
            self.K = self.venv.env_method('get_optimal_K', indices=[0])[0]
        K_optim = torch.nn.Parameter(torch.from_numpy(self.K))
        self.K_optim = torch.nn.ParameterList([K_optim])

    def reset_optimizer(self, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.epoch == 0:
            self.optimizer = None
        else:
            self.optimizer = torch.optim.SGD(
                self.K_optim, self.lr
            )  # works better than Adam
            # self.optimizer = torch.optim.Adam(self.K_optim, self.lr)

    def learn(
        self, tasks=None, policy_path=None, optimizer=None, save_path=None,
        **kwargs
    ):
        if tasks is None:
            tasks = self.task_all
            logging.warning('Re-using tasks!')
        num_task = len(tasks)

        # Reset policy
        self.reset_policy(policy_path, deepcopy(tasks[0]))

        # Reset optimizer
        self.reset_optimizer(optimizer)

        # Run epochs
        reward_avg_all = []
        best_K = np.copy(self.K)
        for epoch_ind in range(self.epoch):

            # Sample tasks
            epoch_tasks_ind = self.rng.choice(
                num_task, size=self.num_episode_per_epoch
            )
            epoch_tasks = [tasks[ind] for ind in epoch_tasks_ind]

            # Set initial conditions and K matrix
            init_x_corr = np.zeros((4, 4))
            for task in epoch_tasks:
                # init_x = self.rng.normal(0, 0.1, size=4)
                # init_x[0] += -np.pi/2
                init_x = np.array(self.init_x)
                task.init_x = init_x.tolist()
                task.Q_gain = self.Q_gain
                task.true_m = [task.m1, task.m2]
                task.true_b = [task.b1, task.b2]
                task.num_step_eval = self.num_step_eval_low  #! use few for training

                init_x_corr += init_x.reshape((4, 1)
                                             ).dot(init_x.reshape((1, 4)))
            init_x_corr /= self.num_episode_per_epoch

            # Run
            self.reset_tasks(epoch_tasks, verbose=False)
            _, reward_all, info_episode = self.run_steps(
                self.num_episode_per_epoch
            )

            # Aggregate gradient calculated in envs
            gradient = np.mean(
                np.concatenate([info['grad'][None] for info in info_episode]),
                axis=0
            )
            # print('Raw Grad: ', gradient)
            # print('Fro Norm of K: ', np.linalg.norm(self.K))

            # Clip gradient to stablize training - check gradient norm carefully
            if np.linalg.norm(gradient) >= self.grad_clip:
                gradient *= (self.grad_clip / np.linalg.norm(gradient))
            # print('Clipped Grad: {} with norm clip {}'.format(gradient, self.grad_clip))

            # Take a step on K
            # print('K: ', self.K)
            self.optimizer.zero_grad()
            self.K_optim[0].grad = torch.from_numpy(gradient)
            self.optimizer.step()
            self.K = self.K_optim[0].clone().detach().numpy()
            # print('K: ', self.K)
            # print()
            # exit()

            # Get avg reward
            reward_avg = np.mean(reward_all.numpy())
            # logging.info('Avg reward: {}'.format(reward_avg))
            reward_avg_all += [reward_avg]

            if epoch_ind == 0 or reward_avg > best_reward:
                best_K = np.copy(self.K)
                best_reward = reward_avg

        # Set policy to best one
        self.K = best_K
        # logging.info('Best K: {}'.format(best_K))
        if self.epoch > 0 and save_path is not None:
            logging.info('Best reward: {}'.format(best_reward))
            plt.plot(reward_avg_all)
            plt.savefig(save_path + '_reward.png')
            plt.close()
            plt.plot(info_episode[0]['x'])
            plt.savefig(save_path + '_sample_traj.png')
            plt.close()
        # plt.show()

        # policy, memory, optimizer
        return deepcopy(self.K), None, deepcopy(self.optimizer)

    def evaluate(
        self, tasks, num_episode=None, policy_path=None, verbose=False
    ):
        """
        """
        # Reset policy
        if policy_path is not None:
            # logging.info('Evaluating using {}'.format(policy_path))
            self.reset_policy(policy_path)
        # else:
        #     logging.warning('Re-using policy for evaluation!')

        # Repeat same task if num_epoisode is larger then task
        if num_episode is None:
            num_episode = len(tasks)
        else:
            assert len(tasks) == 1
            tasks = [tasks[0] for _ in range(num_episode)]

        # Add LQR parameters to tasks -> reset pendulum every time
        for task in tasks:
            task.init_x = self.init_x
            task.Q_gain = self.Q_gain
            task.true_m = [task.m1, task.m2]
            task.true_b = [task.b1, task.b2]
            task.num_step_eval = self.num_step_eval_high  #! use high for evaluation

        # Reset tasks
        self.reset_tasks(tasks, verbose)
        if num_episode is None:
            num_episode = len(tasks)

        # Run episodes - each task exactly once!
        num_episode_run, episodes_reward, episodes_info = self.run_steps(
            num_episode, run_in_seq=True
        )
        assert num_episode_run == num_episode

        # Get avg reward
        eval_reward = torch.sum(episodes_reward) / num_episode_run
        return eval_reward, episodes_info

    def run_steps(
        self,
        num_episode=None,
        run_in_seq=False,
        perturb_policy=False,
    ):

        if run_in_seq:
            task_ids_yet_run = list(np.arange(num_episode))

        cnt = 0
        info_episode = []
        r_episodes = []
        perturb_episode = []
        while cnt < num_episode:

            # Reset
            if run_in_seq:
                assert num_episode == self.num_task
                new_ids = task_ids_yet_run[:self.n_envs]
                task_ids_yet_run = task_ids_yet_run[self.n_envs:]
                self.reset_env_all(new_ids)
            else:
                self.reset_env_all()

            # Apply action
            action = np.tile(self.K[None], reps=(self.n_envs, 1, 1))
            if perturb_policy:
                action += action * (
                    self.rng.random(size=action.shape) - 0.5
                ) * 2 * self.perturb_mag
            s_all, r_all, done_all, info_all = self.step(action)

            # Check all envs
            for env_ind, (s_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):

                # Record
                r_episodes += [r]
                info_episode += [info]

                # Count for eval mode
                cnt += 1

                # Quit
                if cnt == num_episode:
                    break

        # Return reward and info
        r_episodes = torch.cat(r_episodes)
        return cnt, r_episodes, info_episode  #, perturb_episode

    def collect_data(
        self,
        tasks,
        num_traj_per_task=1,
        max_traj_len=20,
        perturb_policy=False,
        traj_type='dp_full',
    ):
        """
        Instead of replying on summarizers, truncate trajectories to max_traj_len here. Concatenate trajectories for the same task.
        """

        # Add parameters to tasks
        for task in tasks:
            task.init_x = [-np.pi, 0, 0, 0]  # same state
            task.Q_gain = self.Q_gain
            task.true_m = [task.m1, task.m2]
            task.true_b = [task.b1, task.b2]
            task.num_step_eval = self.num_step_eval_low  #! use low for data collection

        # Repeat tasks for num_traj_per_task
        num_task = len(tasks)
        num_episode = num_task * num_traj_per_task
        tasks = [item for item in tasks for _ in range(num_traj_per_task)]

        # Run all tasks in sequence
        self.reset_tasks(tasks)
        cnt, r_episodes, info_episode = self.run_steps(
            num_episode=num_episode, run_in_seq=True,
            perturb_policy=perturb_policy
        )

        # Filter to best trajectory
        if 'filter' in traj_type:
            r_episodes_np = r_episodes.clone().cpu().numpy().reshape(
                num_task, num_traj_per_task
            )
            r_episodes_argmax = np.argmax(r_episodes_np, axis=1)

        # Reshape state_all into tasks - filter out unstable ones - only using q right now, no qdot
        if 'final' in traj_type:
            state_all = [
                np.vstack(info['x'][-max_traj_len:, :2])[None]
                for info in info_episode
            ]
        elif 'full' in traj_type:
            state_all = [
                np.vstack(info['x'][:max_traj_len, :2])[None]
                for info in info_episode
            ]
        else:
            raise "Unknown traj type!"
        state_all = torch.from_numpy(np.vstack(state_all)).float().to(
            self.device
        )  # num_episode x max_traj_len x obs_dim

        # Reshape to num_task x num_traj_per_task x max_traj_len x obs_dim
        state_all = state_all.reshape(
            (num_task, num_traj_per_task, max_traj_len, -1)
        )
        if 'filter' in traj_type:
            state_all = state_all[np.arange(state_all.shape[0]), None,
                                  r_episodes_argmax, :, :]  # keep dim

        # Filter unstable tasks - #! issue: if we use multiple goal per task, this would not deal with the different goals. for now we ignore this since for dp linearized, we only use one goal per task
        stable_ind = [
            ind for ind in range(num_task) if torch.max(state_all[ind]) < 1e3
        ]
        state_all = state_all[stable_ind]
        logging.info(
            'Filtered {} unstable tasks'.format(num_task - len(stable_ind))
        )

        # Collect action
        if 'final' in traj_type:
            action_all = [
                np.vstack(info['u'][-max_traj_len:])[None]
                for info in info_episode
            ]
        elif 'full' in traj_type:
            action_all = [
                np.vstack(info['u'][:max_traj_len])[None]
                for info in info_episode
            ]
        action_all = torch.from_numpy(np.vstack(action_all)
                                     ).float().to(self.device)

        # Reshape
        action_all = action_all.reshape(
            (num_task, num_traj_per_task, max_traj_len, -1)
        )
        if 'filter' in traj_type:
            action_all = action_all[np.arange(action_all.shape[0]), None,
                                    r_episodes_argmax, :, :]

        # Filter
        action_all = action_all[stable_ind]

        # Collect parameters
        param_all = torch.from_numpy(
            np.vstack([info['param'] for info in info_episode])
        ).float().to(self.device)
        param_all = param_all.reshape((num_task, num_traj_per_task, -1))
        param_all = param_all[
            stable_ind,
            0, :]  # num_task x param_dim - take the first traj of each task

        return param_all, state_all, action_all, r_episodes

    def reset_tasks(self, tasks, verbose=False):
        self.task_all = tasks
        self.num_task = len(self.task_all)
        if verbose:
            logging.info(f"{self.num_task} tasks are loaded")

    # === Venv ===
    def step(self, action):
        return self.venv.step(action)

    def reset_sim(self):
        raise NotImplementedError

    def reset_env_all(self, task_ids=None, verbose=False):
        if task_ids is None:
            task_ids = self.rng.integers(
                low=0, high=self.num_task, size=(self.n_envs,)
            )

        # fill if not enough
        if len(task_ids) < self.n_envs:
            num_yet_fill = self.n_envs - len(task_ids)
            task_ids += [0 for _ in range(num_yet_fill)]

        tasks = [self.task_all[id] for id in task_ids]
        s = self.venv.reset(tasks)
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    "<-- Reset environment {} with task {}:".format(
                        index, task_ids[index]
                    )
                )
        self.env_step_cnt = [0 for _ in range(self.n_envs)]
        return s, task_ids

    def reset_env(self, env_ind, task_id=None, verbose=False):
        if task_id is None:
            task_id = self.rng.integers(low=0, high=self.num_task)
        s = self.venv.reset_one(env_ind, self.task_all[task_id])
        if verbose:
            logging.info(
                "<-- Reset environment {} with task {}:".format(
                    env_ind, task_id
                )
            )
        self.env_step_cnt[env_ind] = 0
        return s, task_id
