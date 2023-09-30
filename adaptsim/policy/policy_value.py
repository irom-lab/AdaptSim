import os
from copy import deepcopy
from collections import namedtuple
import numpy as np
import torch
import logging

from adaptsim.policy.policy_base import PolicyBase
from adaptsim.policy.replay_memory import ReplayMemoryMeta
from adaptsim.learner import learner_dict
from adaptsim.policy.util import utility_policy_dict
from adaptsim.util.scheduler import StepLRFixed


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])


class PolicyValue(PolicyBase):

    def __init__(self, cfg, venv, verbose=True):
        super().__init__(cfg, venv)

        # Learner
        self.learner_name = cfg.learner.name
        self.learner = learner_dict[self.learner_name](cfg.learner)
        self.learner.build_network(cfg.learner.arch, verbose=verbose)
        self.module_all = [self.learner]  # for saving models
        self.use_append = cfg.learner.arch.append_dim.critic

        # Utility - helper functions for envs
        self.utility = utility_policy_dict[cfg.utility.name](cfg.utility)

        # Parameters - no need to reset
        self.max_sample_steps = cfg.max_sample_steps
        self.batch_size = cfg.batch_size
        self.memory_capacity = cfg.memory_capacity
        self.update_freq = cfg.update_freq
        self.num_update = max(
            1, int(cfg.replay_ratio * self.update_freq / self.batch_size)
        )
        self.check_freq = cfg.check_freq
        self.num_warmup_step_percentage = cfg.num_warmup_step_percentage
        self.num_episode_per_eval = cfg.num_eval_episode
        self.cfg_eps = cfg.eps
        # if hasattr(cfg, 'memory_path') and cfg.memory_path is not None:
        #     self.memory_path = cfg.memory_path
        # if hasattr(cfg, 'optim_path') and cfg.optim_path is not None:
        #     self.reset_optimizer(
        #         torch.load(cfg.optim_path, map_location=self.device)
        #     )
        self.target_eval_reward = cfg.target_eval_reward

    #== Reset policy/optimizer/memory
    def reset_policy(self, policy_path=None):
        if policy_path:
            self.learner.load_network(policy_path)
            # logging.info('Loaded policy network from: {}'.format(policy_path))
        else:
            self.learner.build_network(
                self.cfg.learner.arch, build_optimizer=False, verbose=False
            )
            # logging.info('Built new policy network!')

    def reset_optimizer(self, optimizer_state=None):
        if optimizer_state:
            self.learner.load_optimizer_state(optimizer_state)
            # logging.info('Loaded policy optimizer!')
        else:
            self.learner.build_optimizer()
            # logging.info('Built new policy optimizer!')

    def make_meta_memory(self, memory_path):
        memory = ReplayMemoryMeta(self.memory_capacity, self.seed)
        old_memory = torch.load(memory_path,
                                map_location=self.device)['deque']  # images?
        memory.set_meta_array(old_memory)
        # logging.info(
        #     'Loaded meta memory with meta memory size '
        #     '{}'.format(len(memory.memory_meta))
        # )
        return memory

    #== Learn, evaluate, run steps
    def learn(
        self,
        tasks=None,
        memory=None,
        policy_path=None,
        optimizer_state=None,
        out_folder_postfix=None,
        verbose=False,
        **kwargs,
    ):
        # logging.info('Learning with {} steps!'.format(self.max_sample_steps))
        self.cnt_step = 0

        # Reset tasks
        if tasks is not None:
            self.reset_tasks(tasks, verbose)

        # Reset memory if not provided
        if memory is not None:
            self.memory = memory
        elif hasattr(self, 'memory_path'):
            self.memory = self.make_meta_memory(self.memory_path)
        else:
            self.memory = ReplayMemoryMeta(self.memory_capacity, self.seed)

        # Load NN weights if provided - and also build optimizer if reset
        self.reset_policy(policy_path)

        # Load optimizer state if provided, or reset optimizer
        self.reset_optimizer(optimizer_state)

        # Update out folder if postfix specified
        if out_folder_postfix is not None:
            out_folder = os.path.join(self.out_folder, out_folder_postfix)
        else:
            out_folder = self.out_folder
        self.reset_save_info(out_folder)

        # Exploration
        if hasattr(self.cfg_eps, 'period_percentage'):
            eps_period = int(
                self.max_sample_steps * self.cfg_eps.period_percentage
            )
        else:
            eps_period = self.cfg_eps.period
        self.eps_schduler = StepLRFixed(
            init_value=self.cfg_eps.init, period=eps_period,
            end_value=self.cfg_eps.end, step_size=self.cfg_eps.step
        )

        # Run initial steps
        self.set_train_mode()
        if self.num_warmup_step_percentage > 0:
            num_warmup_step = int(
                self.max_sample_steps * self.num_warmup_step_percentage
            )
            self.cnt_step, _ = self.run_steps(num_step=num_warmup_step)

        # Run rest of steps while optimizing policy
        cnt_opt = 0
        # best_reward = 0
        while self.cnt_step <= self.max_sample_steps:
            print(self.cnt_step, end='\r')

            # Train or eval
            if self.eval_mode:
                num_episode_run, info_episode = self.run_steps(
                    num_episode=self.num_episode_per_eval
                )
                eval_reward_cumulative = \
                    self.eval_reward_cumulative_all / num_episode_run
                self.eval_record[self.cnt_step] = (eval_reward_cumulative,)
                # logging.info(
                #     f'Evaluated at cnt {self.cnt_step} with {eval_reward_cumulative:.3f} success rate'
                # )

                # Saving model (and replay buffer)
                best_path = self.save(metric=eval_reward_cumulative)

                # Save training details
                torch.save({
                    'loss_record': self.loss_record,
                    'eval_record': self.eval_record,
                }, os.path.join(out_folder, 'train_details'))

                # Quite training
                if self.cnt_step == self.max_sample_steps or eval_reward_cumulative > self.target_eval_reward:
                    break

                # Switch to training
                self.set_train_mode()
            else:
                self.cnt_step += self.run_steps(num_step=self.update_freq)[0]

                # Update policy
                loss = 0
                for _ in range(self.num_update):
                    batch_train = self.unpack_batch(self.sample_batch())
                    loss_batch = self.learner.update(batch_train)
                    loss += loss_batch
                loss /= self.num_update

                # Record: loss_q, loss_pi, loss_entropy, loss_alpha
                self.loss_record[self.cnt_step] = loss

                # Count number of optimization
                cnt_opt += 1

                ################### Eval ###################
                if cnt_opt % self.check_freq == 0:
                    self.set_eval_mode()

        ################### Done ###################
        # best_reward = np.max([q[0] for q in self.pq_top_k.queue])  # yikes...
        # logging.info(
        #     'Saving best path {} with reward {:.3f}!'.format(
        #         best_path, best_reward
        #     )
        # )

        # Policy, memory, optimizer
        return best_path, deepcopy(self.memory
                                  ), self.learner.get_optimizer_state()

    def evaluate(
        self,
        tasks=None,
        belief=None,
        num_episode=None,
        policy_path=None,
        verbose=False,
    ):
        # Reset tasks
        if tasks is not None:
            self.reset_tasks(tasks, verbose)
        if num_episode is None:
            num_episode = self.num_task

        # Load NN weights if provided
        self.reset_policy(policy_path)

        # Set current belief for policy
        self.utility.set_belief(belief)

        # Run each task exactly once!
        self.set_eval_mode()
        num_episode_run, episode_info = self.run_steps(
            num_episode=num_episode, run_in_seq=True
        )
        assert num_episode_run == num_episode
        eval_reward_cumulative_avg = self.eval_reward_cumulative_all / num_episode

        # Remove belief
        self.utility.remove_belief()

        if verbose:
            print('Avg reward: ', eval_reward_cumulative_avg)
        return eval_reward_cumulative_avg, episode_info

    def collect_data(
        self,
        tasks,
        num_traj_per_task=1,
        max_traj_len=1,
        traj_type='push_final',
        **kwargs,
    ):
        assert num_traj_per_task == 1, 'Currently only support num_traj_per_task=1'

        # Repeat tasks for num_traj_per_task
        num_task = len(tasks)
        num_episode = num_task * num_traj_per_task
        tasks = [item for item in tasks for _ in range(num_traj_per_task)]

        # Run all tasks in sequence, no perturbing policy
        self.reset_tasks(tasks)
        self.set_eval_mode()
        num_episode_run, info_episode = self.run_steps(
            num_episode=num_episode, run_in_seq=True
        )
        r_episodes = np.array([info['reward'] for info in info_episode]
                             ).reshape(num_task, num_traj_per_task)
        logging.info(f'Collected {num_episode_run} trajectories!')

        # Reshape state_all into tasks - filter out unstable ones
        if traj_type == 'push_final':
            state_all = [
                info['bottle_T_final'][:2][None, None] for info in info_episode
            ]
        elif traj_type == 'push_full':
            state_all = [
                np.vstack(info['bottle_T_all'])[:, :2][None]
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
        elif traj_type == 'scoop':
            state_all = [
                np.vstack(info['veggie_xy_all'])[None] for info in info_episode
            ]
        else:
            raise 'Unknown trajectory type when collecting data!'
        state_all = torch.from_numpy(np.vstack(state_all)).float().to(
            self.device
        )  # num_episode x max_traj_len x obs_dim
        state_all = state_all.reshape(
            (num_task, num_traj_per_task, max_traj_len, -1)
        )

        # Collect action - repeat action for all steps
        action_all = [
            np.tile(info['action'][None, None], (1, max_traj_len, 1))
            for info in info_episode
        ]
        action_all = torch.from_numpy(np.vstack(action_all)
                                     ).float().to(self.device)
        action_all = action_all.reshape(
            (num_task, num_traj_per_task, max_traj_len, -1)
        )

        # Collect parameters
        param_all = torch.from_numpy(
            np.vstack([info['parameter'] for info in info_episode])
        ).float().to(self.device).reshape((num_task, num_traj_per_task, -1))
        param_all = param_all[:, 0, :]  # num_task x param_dim

        return param_all, state_all, action_all, r_episodes

    def run_steps(
        self,
        num_step=None,
        num_episode=None,
        run_in_seq=False,
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

            # Interact
            cur_tasks = [self.task_all[id] for id in task_ids]
            if self.use_append:
                append_all = self.utility.get_append(cur_tasks)
            else:
                append_all = None

            # Select action
            with torch.no_grad():
                a_all = self.forward(s, append=append_all)

            # Apply action - update heading
            s_all, r_all, done_all, info_all = self.step(a_all)

            # Get new append
            append_nxt_all = None

            # Check all envs
            for env_ind, (s_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):
                # Save append
                if append_all is not None:
                    info['append'] = append_all[env_ind].unsqueeze(0)
                if append_nxt_all is not None:
                    info['append_nxt'] = append_nxt_all[env_ind].unsqueeze(0)

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

    def forward(self, state, append):
        noise = not self.eval_mode
        a_all = self.learner(state, append=append, noise=noise).cpu().numpy()
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
        append = None
        if self.use_append:
            append = torch.cat([info['append'] for info in batch.info]
                              ).to(self.device)
        return (state, action, reward, append)
