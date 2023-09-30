import copy
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler

from adaptsim.learner.critic import Critic
from adaptsim.model.branching_q_network import BranchingQNetwork


class BranchingQ(Critic):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.single_step = cfg.single_step
        if not self.single_step:
            self.gamma = cfg.gamma

    def build_network(self, cfg, build_optimizer=True, verbose=True):

        # Bin sizes - can be different sizes
        if isinstance(cfg.num_bin, int):
            cfg.num_bin = [cfg.num_bin for _ in range(cfg.action_dim)]
        self.num_bin = cfg.num_bin

        # Add append_dim to input_dim
        mlp_dim = [cfg.input_dim + cfg.append_dim.critic
                  ] + list(cfg.mlp_dim.critic)
        if not hasattr(cfg, 'use_ln'):
            cfg.use_ln = False
        self.critic = BranchingQNetwork(
            mlp_dim=mlp_dim,
            num_bin=cfg.num_bin,
            skip_dim=cfg.skip_dim,
            activation_type='elu',
            use_ln=cfg.use_ln,
            device=self.device,
            verbose=verbose,
        )

        # Load weights if provided
        if hasattr(cfg, 'critic_path') and cfg.critic_path is not None:
            self.load_network(cfg.critic_path)

        # Copy for critic target
        if not self.single_step:
            self.critic_target = copy.deepcopy(self.critic)

        # Create optimizer
        if not self.eval and build_optimizer:
            logging.info("Build optimizer for inference.")
            self.build_optimizer()

        # Load optimizer if specified
        if 'optimizer_path' in cfg and cfg.optimizer_path is not None:
            self.load_optimizer_state(
                torch.load(cfg.optimizer_path, map_location=self.device)
            )
            logging.info('Loaded optimizer for branching q!')

    def build_optimizer(self):
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_c)
        if self.lr_c_schedule:
            self.critic_scheduler = lr_scheduler.StepLR(
                self.critic_optimizer, step_size=self.lr_c_period,
                gamma=self.lr_c_decay
            )

    def __call__(self, state, append=None, verbose=False):
        out = super().__call__(state, append)
        if verbose:
            logging.info('Branching q with state {}'.format(state))
            logging.info('Branching q values: {}'.format(out))
        if isinstance(out, np.ndarray):
            return np.argmax(out, axis=-1)
        elif isinstance(out, torch.Tensor):
            return torch.argmax(out, dim=-1)  # separate bins
        else:
            logging.error('Critic returns wrong data type - ???')
            raise

    def value(self, state):
        return self.critic.value(state)

    def update(self, batch):

        if not self.single_step:
            state, action, reward, next_state, done, append, next_append = batch
            if append is not None:
                state = torch.hstack((state, append))
            if next_append is not None:
                next_state = torch.hstack((next_state, next_append))

            # Nx(num_branch)
            q_raw = self.critic(state)
            q = q_raw.gather(2, action.unsqueeze(-1)).squeeze(-1)
            # logging.info('q for first two data: {}'.format(q_raw[:2]))
            # logging.info('q with max actions: {}'.format(q))
            with torch.no_grad():
                argmax = torch.argmax(
                    self.critic(next_state), dim=2
                )  # use current network to evaluate action
                # logging.info('argmax: {}'.format(argmax))
                max_next_q = self.critic_target(next_state).gather(
                    2, argmax.unsqueeze(-1)
                ).squeeze(-1)  # use target network to evaluate value
                max_next_q = max_next_q.mean(1, keepdim=True)
            # Nx1
            target_q = reward + max_next_q * self.gamma * (1-done)

            # Repeat target_q for branches!
            target_q = target_q.expand(-1, q.shape[1])

            train_loss = F.mse_loss(target_q, q)
            self.critic_optimizer.zero_grad()
            train_loss.backward()
            # self.critic.scale_gradient()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            # Soft update target every time
            self.update_target_networks()
            return train_loss.item()

        else:
            state, action, reward, append = batch
            if append is not None:
                state = torch.hstack((state, append))

            # Nx|A|
            q = self.critic(state).gather(2, action.unsqueeze(-1)).squeeze(-1)

            # Repeat reward for branch!
            reward = reward.unsqueeze(-1).expand(-1, q.shape[1])

            # Loss
            criterion = torch.nn.SmoothL1Loss()
            train_loss = criterion(q, reward)

            # Update params using clipped gradients
            self.critic_optimizer.zero_grad()
            train_loss.backward()
            # self.critic.scale_gradient()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()
            return train_loss.item()
