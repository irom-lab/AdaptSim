import numpy as np
import torch

from adaptsim.env.util.vec_env import VecEnvBase
from util.numeric import normalize, unnormalize_tanh


class VecEnvDP(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvDP, self).__init__(venv, device)


class VecEnvDPLinearized(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvDPLinearized, self).__init__(venv, device)


class VecEnvDPLinearizedNoise(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvDPLinearizedNoise, self).__init__(venv, device)


class VecEnvAcrobot(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvAcrobot, self).__init__(venv, device)


class VecEnvPendulum(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvPendulum, self).__init__(venv, device)


class VecEnvPush(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvPush, self).__init__(venv, device)

        self.action_lb = np.array([cfg.vel_x_range[0], cfg.yaw_range[0]])
        self.action_ub = np.array([cfg.vel_x_range[1], cfg.yaw_range[1]])
        self.goal_lb = torch.tensor([cfg.goal_x_range[0],
                                     cfg.goal_y_range[0]]).float().to(device)
        self.goal_ub = torch.tensor([cfg.goal_x_range[1],
                                     cfg.goal_y_range[1]]).float().to(device)

    def reset(self, tasks):
        obs = super().reset(tasks)
        return normalize(obs, self.goal_lb, self.goal_ub)

    def reset_one(self, index, task):
        obs = super().reset_one(index, task)
        return normalize(obs, self.goal_lb, self.goal_ub)

    def step(self, actions):
        actions = unnormalize_tanh(actions, self.action_lb, self.action_ub)
        return super().step(actions)

    def get_goal(self):
        """
        Get goal locations from all envs
        """
        return self.get_attr('goal')

    def unnormalize_action(self, actions):
        return unnormalize_tanh(actions, self.action_lb, self.action_ub)

    def sample_random_action(self):
        actions = np.random.uniform(
            -1, 1, size=(self.n_envs, len(self.action_lb))
        )
        return actions


class VecEnvScoop(VecEnvBase):

    def __init__(self, venv, device, cfg):
        super(VecEnvScoop, self).__init__(venv, device)

        self.action_lb = np.array([
            cfg.pitch_range[0] * np.pi / 180,
            cfg.pd1_range[0],
            cfg.s_x_veggie_tip_range[0],
        ])
        self.action_ub = np.array([
            cfg.pitch_range[1] * np.pi / 180,
            cfg.pd1_range[1],
            cfg.s_x_veggie_tip_range[1],
        ])

        self.obs_lb = torch.tensor(cfg.obs_lb).float().to(device)
        self.obs_ub = torch.tensor(cfg.obs_ub).float().to(device)

    def reset(self, tasks):
        obs = super().reset(tasks)
        norm_obs = normalize(obs, self.obs_lb, self.obs_ub)
        return norm_obs

    def reset_one(self, index, task):
        obs = super().reset_one(index, task)
        return normalize(obs, self.obs_lb, self.obs_ub)

    def step(self, actions):
        raw_actions = unnormalize_tanh(actions, self.action_lb, self.action_ub)
        return super().step(raw_actions)

    def unnormalize_action(self, actions):
        return unnormalize_tanh(actions, self.action_lb, self.action_ub)
