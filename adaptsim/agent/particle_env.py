import numpy as np
import logging
import os
from copy import deepcopy

from adaptsim.policy import policy_agent_dict
from adaptsim.env.util.vec_env import VecEnvBase


def assign_val(particles, val_pairs):
    for val_pair in val_pairs:
        name, val_list = val_pair
        if isinstance(val_list, list):
            for p, val in zip(particles, val_list):
                setattr(p, name, val)
        # single value for all particle
        else:
            for p in particles:
                setattr(p, name, deepcopy(val_list))


class VecEnvParticle(VecEnvBase):

    def __init__(
        self,
        venv,
        device='cpu',
        pickle_option='cloudpickle',
        start_method=None,
        fig_folder=None,
    ):
        super(VecEnvParticle,
              self).__init__(venv, device, pickle_option, start_method)
        self.fig_folder = fig_folder

    def step(self, actions):
        return super().step(actions)

    def learn(
        self,
        tasks_all,
        cnt_itr=None,
        adapt_cnt_base=None,
        pid_all=None,
        step=None,
        twin_batch=None,
        **kwargs,
    ):
        """Some args just for naming of the folder and path. Not the best design."""
        if pid_all is not None:
            if adapt_cnt_base is None:
                out_folder_postfix_all = [
                    'itr-{}_pid-{}_target'.format(cnt_itr, pid)
                    for pid in pid_all
                ]
                save_path_all = [
                    os.path.join(
                        self.fig_folder,
                        'itr-{}_pid-{}_target'.format(cnt_itr, pid)
                    ) for pid in pid_all
                ]
            else:
                out_folder_postfix_all = [
                    'itr-{}_cnt-{}_step-{}_pid-{}'.format(
                        cnt_itr, adapt_cnt_base + cnt, step, pid
                    ) for cnt, pid in enumerate(pid_all)
                ]
                save_path_all = [
                    os.path.join(
                        self.fig_folder,
                        'itr-{}_cnt-{}_step-{}_pid-{}_target'.format(
                            cnt_itr, adapt_cnt_base + cnt, step, pid
                        )
                    ) for cnt, pid in enumerate(pid_all)
                ]
        else:
            out_folder_postfix_all = [None for _ in range(self.n_envs)]
            save_path_all = [None for _ in range(self.n_envs)]
        if twin_batch is None:
            twin_batch = [None for _ in range(self.n_envs)]

        train_recv_all = self.env_method_arg(
            'learn',
            [(tasks, out_folder_postfix, save_path, twin)
             for tasks, out_folder_postfix, save_path, twin in
             zip(tasks_all, out_folder_postfix_all, save_path_all, twin_batch)
            ], **kwargs
        )
        return [train_recv[0] for train_recv in train_recv_all], \
                [train_recv[1] for train_recv in train_recv_all], \
                [train_recv[2] for train_recv in train_recv_all]

    def evaluate(self, tasks_all, **kwargs):
        if isinstance(tasks_all[0], list):
            eval_recv_all = self.env_method_arg(
                'evaluate', [(tasks,) for tasks in tasks_all], **kwargs
            )
        else:  # a single task
            eval_recv_all = self.env_method_arg(
                'evaluate', [(tasks_all,) for _ in range(self.n_envs)],
                **kwargs
            )
        return [eval_recv[0] for eval_recv in eval_recv_all], \
                [eval_recv[1] for eval_recv in eval_recv_all], \


class ParticleEnv():

    def __init__(self, cfg_policy, venv, seed):
        self.cfg_policy = cfg_policy
        self.cfg_policy.seed = seed

        # Policy training params
        self.init_max_sample_steps = cfg_policy.init_max_sample_steps
        self.max_sample_step_decay = cfg_policy.max_sample_step_decay
        self.max_sample_step_min = cfg_policy.max_sample_step_min

        # Initialize policy agent
        cfg_policy.max_sample_steps = self.init_max_sample_steps
        self.policy_agent = policy_agent_dict[cfg_policy.name
                                             ](self.cfg_policy, venv)

    def reset(self, particle):
        self.p = particle
        # logging.info('Received particle with id {}'.format(self.p.id))
        return np.array([], dtype=np.single)

    def update_policy_agent_cfg(self):
        itr = self.p.lifetime - 1  # yikes... as we already increment lifetime for the particle before calling learn() here
        # logging.info('lifetime: {}'.format(itr))
        max_sample_steps = max(
            self.init_max_sample_steps - self.max_sample_step_decay * itr,
            self.max_sample_step_min
        )
        self.policy_agent.max_sample_steps = max_sample_steps

    def learn(
        self,
        train_tasks,
        out_folder_postfix,
        save_path,
        twin,
    ):

        if len(train_tasks) == 0:
            assert twin
            logging.info(
                'Copying from twin id {}; no training'.format(twin.id)
            )
            return twin.policy, twin.memory, twin.optim
        else:
            if twin is not None:
                policy = twin.policy
                optim = twin.optim
                memory = twin.memory
                logging.info(
                    'Copying from twin id {}; training'.format(twin.id,)
                )
            else:
                policy = self.p.policy
                optim = self.p.optim
                memory = self.p.memory

            # Update max sample steps based on particle lifetime
            self.update_policy_agent_cfg()

            # Run
            new_policy, new_memory, new_optim = self.policy_agent.learn(
                train_tasks,
                policy_path=policy,
                optimizer_state=optim,
                memory=memory,
                out_folder_postfix=out_folder_postfix,
                save_path=save_path,
                # reuse_policy=reuse_policy
            )
            return new_policy, new_memory, new_optim

    def evaluate(
        self,
        test_tasks,
        #    use_particle_policy=False,
        #    reuse_policy=False
    ):
        meta_reward, meta_info = self.policy_agent.evaluate(
            test_tasks,
            policy_path=self.p.policy,
            # reuse_policy=reuse_policy
        )
        return meta_reward, meta_info

    def evaluate_real(self, tasks_all):
        """
        Actually return information is not used; instead load saved file containing the results."""
        self.policy_agent.evaluate_real(tasks_all, policy_path=self.p.policy)
