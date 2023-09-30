"""
Managing inference network for the simulation parameters.

"""
import os
import numpy as np
import torch
import logging
from collections import namedtuple

from adaptsim.policy.replay_memory import ReplayMemory
from adaptsim.learner.branching_q import BranchingQ


TransitionTask = namedtuple(
    'TransitionTask', ['s', 'a', 'r', 's_', 'done', 'info']
)


class ParamInference():

    def __init__(self, cfg):

        self.device = cfg.device

        # Initialize RNG
        self.rng = np.random.default_rng(seed=cfg.seed)

        # Cfg
        self.num_adapt_param = cfg.num_adapt_param
        self.num_param_bin = cfg.num_param_bin
        self.action_dim = self.num_adapt_param  # mu/std together

        # Build network
        cfg.arch.action_dim = self.action_dim
        cfg.arch.num_bin = cfg.num_param_bin
        cfg.arch.input_dim = self.action_dim + cfg.num_feature  # mean only
        self.network = BranchingQ(cfg)
        self.network.build_network(cfg.arch, verbose=True)
        if hasattr(
            cfg.arch, 'critic_path'
        ) and cfg.arch.critic_path is not None:
            logging.info(
                "Load critic weights from {}".format(cfg.arch.critic_path)
            )

        # Build memory - load experiences if specified
        self.memory = ReplayMemory(cfg.memory_capacity, cfg.seed)
        if hasattr(cfg, 'memory_path') and cfg.memory_path is not None:
            self.memory.set_array(
                torch.load(cfg.memory_path, map_location=self.device)['deque']
            )
            logging.info(
                'Loaded memory size {} for inference agent!'.format(
                    len(self.memory)
                )
            )

    def infer_traj(self, traj):
        if hasattr(self, 'traj_inference'):
            return self.traj_inference(traj)
        else:
            return traj  # e.g., features in DP

    def infer(self, state, random_flags=False, verbose=False):
        """
        Do not infer trajectory here. Assume state includes latent.
        """
        with torch.no_grad():
            a_all = self.network(state, verbose=verbose)  # keep batch dim

        # assign random, assume each param have the same number of bins
        lb = [0] * self.action_dim
        ub = self.num_param_bin
        a_random_all = self.rng.integers(
            lb, ub, size=(len(state), self.action_dim), dtype=int
        )
        a_all = np.where(random_flags, a_random_all, a_all)
        return a_all

    def infer_value(self, state):
        with torch.no_grad():
            value_all = self.network.value(state)
        return value_all

    def save(self, step, logs_path):
        self.network.save(step, logs_path)

        optimizer_path = os.path.join(logs_path, 'optimizer')
        os.makedirs(optimizer_path, exist_ok=True)
        self.network.save_optimizer_state(
            os.path.join(optimizer_path,
                         str(step) + '.pt')
        )

        memory_path = os.path.join(logs_path, 'memory')
        os.makedirs(memory_path, exist_ok=True)
        self.memory.save(os.path.join(memory_path, str(step) + '.pt'))

    # === Replay and update ===
    def update_inference(self, batch_size):
        batch_raw = self.sample_batch(batch_size)
        batch = self.unpack_batch(batch_raw)
        loss = self.network.update(batch)
        return loss

    def sample_batch(self, batch_size=None, recent_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        transitions, _ = self.memory.sample(batch_size, recent_size)
        batch = TransitionTask(*zip(*transitions))
        return batch

    def unpack_batch(self, batch):
        state = torch.from_numpy(np.vstack(batch.s)).float().to(self.device)
        next_state = torch.from_numpy(np.vstack(batch.s_)
                                     ).float().to(self.device)
        reward = torch.FloatTensor(batch.r).unsqueeze(-1).to(self.device)
        action = torch.from_numpy(np.vstack(batch.a)).long().to(self.device)
        done = torch.FloatTensor(batch.done).unsqueeze(-1).to(self.device)
        append = None
        next_append = None
        return state, action, reward, next_state, done, append, next_append

    def store_transition(self, *args):
        self.memory.update(TransitionTask(*args))
