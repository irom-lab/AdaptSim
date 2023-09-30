"""
Base class for managing the simulation parameters.

"""
import numpy as np
from copy import deepcopy
import logging
from omegaconf import OmegaConf

from adaptsim.util.numeric import sample_uniform, normalize
from adaptsim.util.dist import Uniform, Gaussian


class ParameterBase():

    def __init__(self, cfg):

        super().__init__()
        self.device = cfg.device
        if hasattr(cfg, 'num_param_bin_for_continuous'):
            self.num_param_bin_for_continuous = cfg.num_param_bin_for_continuous
        if hasattr(cfg, 'sample_ood_full_width_fraction'):
            self.sample_ood_full_width_fraction = cfg.sample_ood_full_width_fraction

        # Initialize RNG
        self.rng = np.random.default_rng(seed=cfg.seed)

        # Initialize params
        self.initialize(cfg.param)

    @property
    def dist_type(self):
        return self._dist_type

    # Right now only support same type for all parameters. Ignore discrete too.

    @property
    def adapt_param_cfg(self):
        return deepcopy(self._adapt_param_cfg)

    @property
    def fixed_param_cfg(self):
        return self._fixed_param_cfg

    @property
    def param_cfg(self):
        return OmegaConf.merge(self._adapt_param_cfg, self._fixed_param_cfg)

    @property
    def num_param_bin(self):
        return self._num_param_bin

    @property
    def num_adapt_param(self):
        return self._num_adapt_param

    @property
    def adapt_param_name(self):
        return self._adapt_param_name

    @property
    def lower_bound(self):
        return np.array([cfg.minimum for _, cfg in self.param_cfg.items()])

    @property
    def upper_bound(self):
        return np.array([cfg.maximum for _, cfg in self.param_cfg.items()])

    @property
    def adapt_param_lower_bound(self):
        return np.array([
            cfg.minimum for _, cfg in self._adapt_param_cfg.items()
        ])

    @property
    def adapt_param_upper_bound(self):
        return np.array([
            cfg.maximum for _, cfg in self._adapt_param_cfg.items()
        ])

    @property
    def adapt_param_mean(self):
        return np.array([(cfg.maximum - cfg.minimum) / 2 + cfg.minimum
                         for _, cfg in self._adapt_param_cfg.items()])

    @property
    def adapt_param_range(self):
        return np.array([
            cfg.maximum - cfg.minimum
            for _, cfg in self._adapt_param_cfg.items()
        ])

    def get_adapt_task_string(self, task):
        """
        Get vector representation of task (mixed with string if discrete parameter). Only for debugging purposes. ALWAYS USE NORMALIZED VERSION FOR TRAINING.
        """
        out = ''
        for name in self.adapt_param_name:
            name = name.lower()
            out += name + '-' + str(task[name]) + '_'
        return out

    def generate_task_wid(self, num_task, particle, clip_param=True, **kwargs):
        """
        Generate tasks by sampling from the parameter distribution
        """
        tasks = []
        for task_id in range(num_task):
            task = OmegaConf.create()
            task.id = task_id

            # Sample for the task - assume in the order of adapt then fixed
            sample = particle.dist.gen(self.rng, n_samples=1)[0]
            if particle.fixed_dist is not None:
                sample_fixed = particle.fixed_dist.gen(self.rng,
                                                       n_samples=1)[0]
                sample = np.hstack((sample, sample_fixed))
            assert len(sample) == len(self.param_cfg)

            # Clip if specified
            if clip_param:
                sample = np.clip(sample, self.lower_bound, self.upper_bound)

            for value, name in zip(sample, self.param_cfg.keys()):
                task[name.lower()] = float(value)

            tasks += [task]
        return tasks

    def generate_task_ood(self, num_task, particle, clip_param=True, **kwargs):
        """
        Generate tasks by sampling from the full min/max. Fixed param also have all min and max.
        Note the sample can still be wid.
        """
        adapt_param_full_width = self.adapt_param_upper_bound - self.adapt_param_lower_bound

        tasks = []
        for task_id in range(num_task):
            task = OmegaConf.create()
            task.id = task_id

            # Sample for the task - assume in the order of adapt then fixed
            while 1:
                sample = []
                # assume in the order!
                for name, param_cfg in self.adapt_param_cfg.items():
                    value = sample_uniform(
                        self.rng, [param_cfg.minimum, param_cfg.maximum]
                    )
                    sample += [value]
                sample = np.array(sample)

                # Rejection sampling
                diff = np.abs(sample - particle.dist.mean)
                if np.all(
                    diff < (
                        adapt_param_full_width
                        * self.sample_ood_full_width_fraction + 1e-6
                    )
                ):
                    break

            # cases when sampling outside the fixed dist? not necessary right now
            if particle.fixed_dist is not None:
                sample_fixed = particle.fixed_dist.gen(self.rng,
                                                       n_samples=1)[0]
                sample = np.hstack((sample, sample_fixed))
            assert len(sample) == len(self.param_cfg)

            # Clip if specified
            if clip_param:
                sample = np.clip(sample, self.lower_bound, self.upper_bound)

            # sample = np.array([4.0, 6.0, 2.0, 8.0])

            # Assign
            for value, name in zip(sample, self.param_cfg.keys()):

                # Check discrete vs continuous
                # if isinstance(param, str):
                #     task[name.lower()] = self.rng.choice(param)

                # else:
                task[name.lower()] = float(value)

            tasks += [task]
        return tasks

    def initialize(self, cfg):
        """
        Generate three cfg - adapt_param, fixed_param, and adapt_param_cfg (name, step, min, max). All of them have name as entries
        """
        self._num_adapt_param = 0
        self._adapt_param_cfg = OmegaConf.create()
        self._fixed_param_cfg = OmegaConf.create()
        self._num_param_bin = []

        # Go through all params
        for name, param in cfg.items():

            # discrete
            if param.dist == 'discrete' and param.fixed:

                self._fixed_param_cfg[name] = OmegaConf.create()
                self._fixed_param_cfg[name].action_all = param.choice  #?
                self._fixed_param_cfg[name].dist = 'discrete'
                # min, max

            elif param.dist == 'discrete' and not param.fixed:

                # Copy to adapt_param_cfg
                self._adapt_param_cfg[name] = OmegaConf.create()
                self._adapt_param_cfg[name].action_all = param.choice
                self._adapt_param_cfg[name].dist = 'discrete'

                # Count number of params to be adapted
                self._num_adapt_param += 1

                # Count bins
                self._num_param_bin += [len(param.choices)]

            # continuous adapt
            elif not param.fixed:
                self._adapt_param_cfg[name] = OmegaConf.create()
                self._adapt_param_cfg[name].dist = param.dist

                self._adapt_param_cfg[name].step = param.step
                self._adapt_param_cfg[name].minimum = param.minimum
                self._adapt_param_cfg[name].maximum = param.maximum

                if param.dist == 'gaussian':
                    self._adapt_param_cfg[name].init_std = param.init_std
                    self._adapt_param_cfg[name].min_std = param.min_std
                    self._adapt_param_cfg[name].max_std = param.max_std
                    self._dist_type = 'gaussian'
                else:
                    self._adapt_param_cfg[name].init_width = param.init_width
                    self._dist_type = 'uniform'

                if 'discrete' in param and param.discrete:
                    self._adapt_param_cfg[name].discrete = True
                else:
                    self._adapt_param_cfg[name].discrete = False

                # Count number of params to be adapted
                self._num_adapt_param += 1

                # Count bins - both mu and std - right now does not support different bin sizes
                self._num_param_bin += [self.num_param_bin_for_continuous]
                # self.num_param_bin_for_continuous]

            # continuous fixed
            else:
                self._fixed_param_cfg[name] = OmegaConf.create()
                self._fixed_param_cfg[name].dist = param.dist

                self._fixed_param_cfg[name].minimum = float(param.minimum)
                self._fixed_param_cfg[name].maximum = float(param.maximum)
                if param.dist == 'gaussian':  # mean, init_std, min, max
                    self._fixed_param_cfg[name].mean = float(param.mean)
                    self._fixed_param_cfg[name].init_std = float(
                        param.init_std
                    )
                else:  # uniform - range, min, max
                    self._fixed_param_cfg[name].range = [
                        float(param.range[0]),
                        float(param.range[1])
                    ]

        self._adapt_param_name = [
            name for name in self._adapt_param_cfg.keys()
        ]

        # Debug
        logging.info('Loaded param type: {}'.format(self.dist_type))
        logging.info(
            'Loaded number of adapt params: {}'.format(self.num_adapt_param)
        )
        logging.info(
            'Loaded adapt param names: {}'.format(self.adapt_param_name)
        )
        logging.info(
            'Loaded adapt param cfg: {}'.format(self._adapt_param_cfg)
        )
        logging.info(
            'Loaded fixed param cfg: {}'.format(self._fixed_param_cfg)
        )
        logging.info('Loaded param bins: {}'.format(self._num_param_bin))

    def apply_action(self, a_all, particle):
        """
        Infer changes in parameter with input of (1) prior distribution (2) trajectory/reward. Update the distribution internally based on NN output
        """
        # Get parameters of interest from particle
        if self.dist_type == 'gaussian':
            new_mean = np.zeros((self.num_adapt_param))
            new_std = np.zeros((self.num_adapt_param))
        elif self.dist_type == 'uniform':
            new_lb = np.zeros((self.num_adapt_param))
            new_ub = np.zeros((self.num_adapt_param))
        else:
            raise

        # Fold into params
        # a_all = a_all.reshape((-1, 2))

        # Apply to all params
        for ind, (param_cfg, a_ind) in enumerate(
            zip(self._adapt_param_cfg.values(), a_all)
        ):
            minimum = param_cfg.minimum
            maximum = param_cfg.maximum
            full_width = maximum - minimum

            if self.dist_type == 'discrete':
                choice = param_cfg.action_all[int(a_ind)]
                # particle.adapt_param[name] = choice
                raise 'Adapt for discrete parameter not supported yet!'

            elif self.dist_type == 'uniform':
                step = param_cfg.step
                half_step = step / 2
                action_all = [
                    [-full_width * step, -full_width * step],
                    [full_width * step, full_width * step],
                    [-full_width * half_step,
                     full_width * half_step],  # no expand for now
                    [full_width * half_step, -full_width * half_step],
                    [0, 0],  # no change
                ]

                # Get action from preset
                lb_action, ub_action = action_all[int(a_ind)]

                # Get current lb/ub
                lb = particle.dist.lb[ind]
                ub = particle.dist.ub[ind]

                # Update
                lb += lb_action
                ub += ub_action

                # Clip parmater - first minimum then maximum
                lb = max(minimum, lb)
                ub = max(minimum, ub)
                lb = min(maximum, lb)
                ub = min(maximum, ub)

                # Clip parameter - ub > lb
                ub = max(ub, lb)

                # Assign
                new_lb[ind] = lb
                new_ub[ind] = ub

            elif self.dist_type == 'gaussian':
                step = param_cfg.step
                action_all = [
                    [-full_width * step, 1],
                    #   [-full_width*step, 0.5],
                    #   [-full_width*step, 2],
                    [full_width * step, 1],
                    #   [full_width*step, 0.5],
                    #   [full_width*step, 2],
                    [0, 1],
                    #   [0, 0.5],
                    #   [0, 2],
                ]
                # mean_action_all = [-full_width*step, full_width*step, 0]
                # std_action_all = [1.5, 1, 0.5]

                # Get action from preset
                # mean_action = mean_action_all[int(a_ind[0])]
                # std_action = std_action_all[int(a_ind[1])]
                mean_action, std_action = action_all[int(a_ind)]

                # Get current mean/std
                mean = particle.dist.mean[ind]
                std = particle.dist.diag_std[ind]

                # Update
                mean += mean_action
                std *= std_action

                # Clip parmater - first minimum then maximum
                mean = max(minimum, mean)
                mean = min(maximum, mean)

                # Round mean if discrete
                if param_cfg.discrete:
                    mean = round(mean)

                # Clip std
                std = max(param_cfg.min_std, std)
                std = min(param_cfg.max_std, std)

                # Assign
                new_mean[ind] = mean
                new_std[ind] = std

            else:
                raise

        # Make a new distribution
        if self.dist_type == 'gaussian':
            particle.dist = Gaussian(m=new_mean, L=new_std)
        elif self.dist_type == 'uniform':
            particle.dist = Uniform(lb_array=new_lb, ub_array=new_ub)

    def get_state_representation(self, particle):
        """
        Get vector representation of adapt param as state in adaptation.
        Discrete: one-hot
        Uniform: lb/ub both normalized to min/max
        Gaussian: mean normalized to min/max and diagonal std
        """
        vec = []
        for ind, (name, param_cfg) in enumerate(self._adapt_param_cfg.items()):
            minimum = param_cfg.minimum
            maximum = param_cfg.maximum

            # Use one-hot for discrete
            if param_cfg.dist == 'discrete':
                raise
                # vec += [param_cfg.action_all.index(param)]

            elif param_cfg.dist == 'gaussian':
                mean = particle.dist.mean[ind]
                if param_cfg.discrete:
                    mean = round(mean)
                std = particle.dist.diag_std[ind]
                mean_norm = normalize(mean, minimum, maximum)
                # std_norm = normalize(std, param_cfg.min_std, param_cfg.max_std)
                # vec += [mean_norm, std_norm]
                vec += [mean_norm]  # TODO: use std too
            else:
                lb = particle.dist.lb[ind]
                ub = particle.dist.ub[ind]
                lb_norm = normalize(lb, minimum, maximum)
                ub_norm = normalize(ub, minimum, maximum)
                vec += [lb_norm, ub_norm]
        return np.array(vec)[None]
