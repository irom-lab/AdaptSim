from abc import ABC, abstractmethod
import os
import torch
import logging

from .utils import soft_update, save_model


class Critic(ABC):

    def __init__(self, cfg):
        self.cfg = cfg
        self.eval = cfg.eval
        self.device = cfg.device

        # == PARAM FOR TRAINING ==
        if not self.eval:

            # Learning Rate
            self.lr_c = cfg.lr_c
            self.lr_c_schedule = cfg.lr_c_schedule
            if self.lr_c_schedule:
                self.lr_c_period = cfg.lr_c_period
                self.lr_c_decay = cfg.lr_c_decay
                self.lr_c_end = cfg.lr_c_end

            # Target Network Update TODO
            self.tau = cfg.tau

    def __call__(self, state, append=None):
        if state is not None:
            x = state
            if append is not None:
                x = torch.hstack((x, append))
        else:
            if append is not None:
                x = append
            else:
                logging.error('Empty state and append!')
                raise
        return self.critic(x)

    @property
    def parameters(self):
        return self.critic.parameters()

    @abstractmethod
    def build_network(self, cfg_arch, verbose=True):
        raise NotImplementedError

    @abstractmethod
    def update(self, batch):
        raise NotImplementedError

    def value(self, x):
        raise NotImplementedError

    def get_optimizer_state(self):
        return self.critic_optimizer.state_dict()

    def save_optimizer_state(self, path):
        torch.save(self.critic_optimizer.state_dict(), f=path)

    def load_optimizer_state(self, optimizer_state):
        self.critic_optimizer.load_state_dict(optimizer_state)

    def load_network(self, path):
        self.critic.load_state_dict(torch.load(path, map_location=self.device))

    def update_critic_hyperParam(self):
        if self.lr_c_schedule:
            lr = self.critic_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.lr_c_end:
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.lr_c_end
            else:
                self.critic_scheduler.step()

    def update_hyper_param(self):
        self.update_critic_hyperParam()

    def update_target_networks(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def save(self, step, logs_path, max_model=None):
        path_c = os.path.join(logs_path, 'critic')
        return save_model(self.critic, path_c, 'critic', step, max_model)

    def remove(self, step, logs_path):
        path_c = os.path.join(
            logs_path, 'critic', 'critic-{}.pth'.format(step)
        )
        # logging.info("Remove {}".format(path_c))
        if os.path.exists(path_c):
            os.remove(path_c)
