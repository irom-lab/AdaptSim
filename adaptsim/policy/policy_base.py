import os
from queue import PriorityQueue
import logging
import numpy as np
import pickle


class PolicyBase():

    def __init__(self, cfg, venv):

        # Params
        self.venv = venv
        self.cfg = cfg
        self.seed = cfg.seed
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.device = cfg.device
        self.eval = cfg.eval

        # Params for vectorized envs
        self.n_envs = cfg.num_env
        self.action_dim = cfg.action_dim
        self.max_steps_train = cfg.max_steps_train
        self.max_steps_eval = cfg.max_steps_eval
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        # Params for saving
        self.save_top_k = cfg.save_top_k
        self.out_folder = cfg.out_folder
        self.reset_save_info(self.out_folder)

        # Load tasks
        if hasattr(cfg, 'dataset'):
            logging.info("= loading tasks from {}".format(cfg.dataset))
            with open(cfg.dataset, 'rb') as f:
                task_all = pickle.load(f)
            self.reset_tasks(task_all)
        else:
            logging.warning('= No dataset loaded!')

    def reset_save_info(self, out_folder):
        os.makedirs(out_folder, exist_ok=True)
        self.module_folder_all = [out_folder]
        self.pq_top_k = PriorityQueue()

        # Save memory
        self.memory_folder = os.path.join(out_folder, 'memory')
        os.makedirs(self.memory_folder, exist_ok=True)
        self.optim_folder = os.path.join(out_folder, 'optim')
        os.makedirs(self.optim_folder, exist_ok=True)

        # Save loss and eval info, key is step number
        self.loss_record = {}
        self.eval_record = {}

    def reset_tasks(self, tasks, verbose=True):
        self.task_all = tasks
        self.num_task = len(self.task_all)
        if verbose:
            logging.info(f"{self.num_task} tasks are loaded")

    def set_train_mode(self):
        self.eval_mode = False
        self.max_env_step = self.max_steps_train

    def set_eval_mode(self):
        self.eval_reward_cumulative = [
            0 for _ in range(self.n_envs)
        ]  # for calculating cumulative reward
        self.eval_reward_best = [0 for _ in range(self.n_envs)]
        self.eval_reward_cumulative_all = 0
        self.eval_reward_best_all = 0
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        self.eval_mode = True
        self.max_env_step = self.max_steps_eval

    # === Venv ===
    def step(self, action):
        return self.venv.step(action)

    def reset_sim(self):
        self.venv.env_method('close_pb')

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
        task = self.task_all[task_id]
        s = self.venv.reset_one(index=env_ind, task=task)
        if verbose:
            logging.info(
                "<-- Reset environment {} with task {}:".format(env_ind, task)
            )
        self.env_step_cnt[env_ind] = 0
        return s, task_id

    # === Models ===
    def save(self, metric=0, force_save=False):
        assert metric is not None or force_save, \
            "should provide metric of force save"
        save_current = True
        if force_save or self.pq_top_k.qsize() < self.save_top_k:
            self.pq_top_k.put((metric, self.cnt_step))
        elif metric > self.pq_top_k.queue[0][
            0]:  # overwrite entry with lowest metric (index=0)
            # Remove old one
            _, step_remove = self.pq_top_k.get()
            for module, module_folder in zip(
                self.module_all, self.module_folder_all
            ):
                module.remove(int(step_remove), module_folder)
            self.pq_top_k.put((metric, self.cnt_step))
        else:
            save_current = False

        if save_current:
            for module, module_folder in zip(
                self.module_all, self.module_folder_all
            ):
                path = module.save(self.cnt_step, module_folder)

        # always return the best path!  # todo minor: fix hack
        return os.path.join(
            self.module_folder_all[0], 'critic',
            'critic-{}.pth'.format(self.pq_top_k.queue[-1][1])
        )
