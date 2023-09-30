import copy
import logging
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler

from adaptsim.learner.critic import Critic
from adaptsim.model.naf_network import NAF_Network


class NAF(Critic):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.single_step = cfg.single_step
        self.cnt = 0

    def build_network(self, cfg, build_optimizer=True, verbose=True):

        # Add append_dim to input_dim
        mlp_dim = [cfg.input_dim + cfg.append_dim.critic
                  ] + list(cfg.mlp_dim.critic)
        self.critic = NAF_Network(
            seed=self.cfg.seed,
            mlp_dim=mlp_dim,
            noise_scale=self.cfg.noise_scale,
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
                self.critic_optimizer,
                step_size=self.lr_c_period,
                gamma=self.lr_c_decay,
            )

    def __call__(self, state, append=None, noise=False, verbose=False):
        out = self.critic.get_action(state, append, noise)
        if verbose:
            logging.info('Branching q with state {}'.format(state))
        return out

    def update(self, batch):

        state, action, reward, append = batch
        if append is not None:
            state = torch.hstack((state, append))

        # Get expected Q values from local model
        q, _ = self.critic.get_value(state, action)

        # Repeat reward for branch!
        reward = reward.unsqueeze(-1)

        # Loss
        criterion = torch.nn.SmoothL1Loss()
        train_loss = criterion(q, reward)

        # Update params using clipped gradients
        self.critic_optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        return train_loss.item()
