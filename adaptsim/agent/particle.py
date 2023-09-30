from dataclasses import dataclass
from scipy.stats import qmc
from omegaconf import OmegaConf
import numpy as np
from copy import deepcopy

from adaptsim.util.dist import Gaussian, Uniform


def find_twin_batch(
    p_batch, p_all, threshold, param_range, lifetime_threshold
):
    """Look for particle from p_all that is close to p. Only looking at the mean right now.
    """
    mean_all = np.vstack([p.dist.mean for p in p_all])
    twin_batch = []
    flag_train_batch = []
    for p in p_batch:
        diff = np.abs(
            p.dist.mean - mean_all
        ) / param_range  # effectively normalized
        diff = np.linalg.norm(diff, axis=1)
        twin_ind_all = np.where(diff < threshold)[0]
        twin_argsort = np.argsort(diff[twin_ind_all])  # low diff first
        twin_ind_all = twin_ind_all[twin_argsort]

        # No need to exclude same id since this particle has not been added to p_all
        # diff_id_all = [p_all[ind].id for ind, val in enumerate(diff_ind_all) if val]
        # logging.info('diff ids {} cur id {}'.format(diff_id_all, p.id))
        # same_ind = np.where(diff_id_all==p.id)[0][0]   # assume only one
        # diff_ind_all = diff_ind_all[diff_ind_all != same_ind]

        # Choose the closest one that satisfies the lifetime threshold and not a twin
        # already - also skip particle without policy
        flag = False
        flag_train = True
        for twin_ind in twin_ind_all:
            p_twin = p_all[twin_ind]

            if not p_twin.twin and p_twin.policy:

                # do not train if policy already been a while
                if p_twin.lifetime >= lifetime_threshold:
                    twin_batch += [p_twin]
                    flag = True
                    flag_train = False
                    break

        if not flag:  # no twin
            twin_batch += [None]
            flag_train_batch += [True]
        else:  # use twin
            flag_train_batch += [flag_train]
    return twin_batch, flag_train_batch


@dataclass
class Particle:
    id: int
    reward: float
    pre_reward: float  # not really used
    info: dict
    policy: str  # path only
    memory: ...
    optim: ...
    dist: ...
    fixed_dist: ...
    lifetime: int  # number of adaptations it has gone through - determining policy training cfg such as max_sample_steps
    twin: bool  # whether this particle already uses policy from a different one; we don't want to borrowing policies twice across particles
    num_clone: int  # number of times used for other particle's policy - prohibit cloning if exceeds some threshold
    just_spawned: bool  # if just spawned, do not adapt at the same iteration since its info is not correct

    def __str__(self):
        return 'id: {}; dist: {}; fixed dist: {}; reward: {}; pre_reward: {}; policy: {}; lifetime: {}'.format(
            self.id,
            self.dist,
            self.fixed_dist,
            self.reward,
            self.pre_reward,
            self.policy,
            self.lifetime,
        )

    @property
    def mean(self):
        mean = self.dist.mean
        if self.fixed_dist is not None:
            mean = np.concatenate((mean, self.fixed_dist.mean))
        return mean

    @property
    def left_param(self):
        if isinstance(self.dist, Uniform):
            return self.dist.lb
        else:
            return self.dist.mean

    @property
    def right_param(self):
        if isinstance(self.dist, Uniform):
            return self.dist.ub
        else:
            return self.dist.diag_std


class ParticleFactory():

    def __init__(self, seed, adapt_param_cfg, fixed_param_cfg):
        # Initialize RNG
        self.rng = np.random.default_rng(seed=seed)

        self.adapt_param_cfg = adapt_param_cfg
        self.fixed_param_cfg = fixed_param_cfg
        self.adapt_param_range = np.array([
            cfg.maximum - cfg.minimum
            for _, cfg in self.adapt_param_cfg.items()
        ])
        self.param_cfg = OmegaConf.merge(
            self.adapt_param_cfg, self.fixed_param_cfg
        )
        self.num_adapt_param = len(self.adapt_param_cfg)
        self.l_bounds = []
        self.u_bounds = []
        for _, param_cfg in self.adapt_param_cfg.items():
            self.l_bounds += [param_cfg.minimum]
            self.u_bounds += [param_cfg.maximum]
        self.l_bounds = np.array(self.l_bounds)
        self.u_bounds = np.array(self.u_bounds)

    def clone_particle(self, p, perturbation=0, p_id=None):
        """Always set twin=True"""
        p_copy = deepcopy(p)
        if perturbation > 0:
            while 1:  # rejection sampling
                # perturbation_base = (self.rng.random(size=self.num_adapt_param)-0.5)*2  # [-1,1]

                # sample randomly from [-1, -0.5] and [0.5, 1]
                perturbation_base = self.rng.choice(
                    [-1, 1], size=self.num_adapt_param
                ) * (self.rng.random(size=self.num_adapt_param) * 0.5 + 0.5)
                new_mean = p_copy.dist.mean + perturbation * perturbation_base * self.adapt_param_range
                if np.all(new_mean > self.l_bounds
                         ) and np.all(new_mean < self.u_bounds):
                    break
            p_copy.dist.mean = new_mean
        if p_id is not None:
            p_copy.id = p_id
        p_copy.twin = True
        return p_copy

    def gen_particle(self, params, p_id):
        """Only for meta particle rn"""
        lp_all = []
        rp_all = []
        lp_fixed_all = []
        rp_fixed_all = []
        for name, param in params.items():

            if name in self.adapt_param_cfg:
                if param.dist == 'uniform':
                    dist_type = 'uniform'
                    lp_all += [param.range[0]]
                    rp_all += [param.range[1]]
                elif param.dist == 'gaussian':
                    dist_type = 'gaussian'
                    lp_all += [param.mean]
                    rp_all += [param.std]
                else:
                    raise
                    # param_interest[name] = param.choice

            elif name in self.fixed_param_cfg:
                if param.dist == 'uniform':
                    fixed_dist_type = 'uniform'
                    lp_fixed_all += [param.range[0]]
                    rp_fixed_all += [param.range[1]]
                elif param.dist == 'gaussian':
                    fixed_dist_type = 'gaussian'
                    lp_fixed_all += [param.mean]
                    rp_fixed_all += [param.std]
                else:
                    raise

            else:
                raise

        lp_all = np.array(lp_all)
        rp_all = np.array(rp_all)
        lp_fixed_all = np.array(lp_fixed_all)
        rp_fixed_all = np.array(rp_fixed_all)

        # Initialize distribution and particle
        if dist_type == 'uniform':
            dist = Uniform(lb_array=lp_all, ub_array=rp_all)
        elif dist_type == 'gaussian':
            dist = Gaussian(m=lp_all, L=rp_all)

        if len(lp_fixed_all) > 0 and fixed_dist_type == 'uniform':
            fixed_dist = Uniform(lb_array=lp_fixed_all, ub_array=rp_fixed_all)
        elif len(lp_fixed_all) > 0 and fixed_dist_type == 'gaussian':
            fixed_dist = Gaussian(m=lp_fixed_all, L=rp_fixed_all)
        else:
            fixed_dist = None

        particle = Particle(
            dist=dist,
            fixed_dist=fixed_dist,
            reward=0,
            pre_reward=0,
            info={},
            policy='',
            memory=None,
            optim=None,
            id=p_id,
            lifetime=0,
            twin=False,
            num_clone=0,
            just_spawned=False,
        )
        return particle

    def gen_space_filling_particle(
        self,
        num_particle,
        initial_policy,
        initial_memory,
        initial_optim,
        use_init_std=True,
    ):

        # Sample means of adapt params with LHC - scale within min/max - take into account the initial width of the distribution
        adapt_param_mean_sampler = qmc.LatinHypercube(d=self.num_adapt_param)
        adapt_param_mean_all = adapt_param_mean_sampler.random(n=num_particle)
        l_bounds = []
        u_bounds = []
        for _, param_cfg in self.adapt_param_cfg.items():
            if num_particle == 1:  # use mean if only one particle
                full_width = param_cfg.maximum - param_cfg.minimum
                l_bounds += [param_cfg.minimum + full_width/2 - 1e-6]
                u_bounds += [param_cfg.maximum - full_width/2 + 1e-6]
            elif param_cfg.dist == 'uniform':
                l_bounds += [param_cfg.minimum + param_cfg.init_width / 2]
                u_bounds += [param_cfg.maximum - param_cfg.init_width / 2]
            elif param_cfg.dist == 'gaussian':
                l_bounds += [param_cfg.minimum + param_cfg.init_std / 2]
                u_bounds += [param_cfg.maximum - param_cfg.init_std / 2]
            else:
                l_bounds += [0]
                u_bounds += [len(param_cfg.action_all)]
                raise
        particle_all_param_mean = qmc.scale(
            adapt_param_mean_all, l_bounds, u_bounds
        )

        # Set range based on sampled means and width in cfg
        p_all = []
        for p_ind, particle_param_mean in enumerate(particle_all_param_mean):
            memory = deepcopy(initial_memory)
            optim = deepcopy(initial_optim)
            lp_all = []
            rp_all = []

            for param_mean, (_, param_cfg) in zip(
                particle_param_mean, self.adapt_param_cfg.items()
            ):
                if param_cfg.dist == 'uniform':
                    lp_all += [param_mean - param_cfg.init_width / 2]
                    rp_all += [param_mean + param_cfg.init_width / 2]
                    dist_type = 'uniform'
                elif param_cfg.dist == 'gaussian':
                    if param_cfg.discrete:
                        param_mean = round(param_mean)
                    lp_all += [param_mean]
                    if use_init_std:
                        rp_all += [param_cfg.init_std]
                    else:  # gaussian around init_std
                        std_sample = self.rng.normal(
                            loc=param_cfg.init_std,
                            scale=(param_cfg.max_std - param_cfg.min_std) / 4
                        )
                        rp_all += [
                            np.clip(
                                std_sample, param_cfg.min_std,
                                param_cfg.max_std
                            )
                        ]
                    dist_type = 'gaussian'

            lp_all = np.array(lp_all)
            rp_all = np.array(rp_all)

            # Initialize adapt dist
            if dist_type == 'gaussian':
                dist = Gaussian(m=lp_all, L=rp_all)
            elif dist_type == 'uniform':
                dist = Uniform(lb_array=lp_all, ub_array=rp_all)
            else:
                raise

            # Copy over fixed param
            lp_fixed_all = []
            rp_fixed_all = []
            for name, param_cfg in self.fixed_param_cfg.items():

                if param_cfg.dist == 'uniform':
                    lp_fixed_all += [param_cfg.range[0]]
                    rp_fixed_all += [param_cfg.range[1]]
                    fixed_dist_type = 'uniform'

                elif param_cfg.dist == 'gaussian':
                    lp_fixed_all += [param_cfg.mean]
                    rp_fixed_all += [param_cfg.init_std]
                    fixed_dist_type = 'gaussian'

                else:
                    action_ind = int(param_mean)  # round down
                    param = [param_cfg.action_all[action_ind]]
                    raise

            lp_fixed_all = np.array(lp_fixed_all)
            rp_fixed_all = np.array(rp_fixed_all)

            # Initialize fixed dist
            if len(lp_fixed_all) > 0 and fixed_dist_type == 'gaussian':
                fixed_dist = Gaussian(m=lp_fixed_all, L=rp_fixed_all)
            elif len(lp_fixed_all) > 0 and fixed_dist_type == 'uniform':
                fixed_dist = Uniform(
                    lb_array=lp_fixed_all, ub_array=rp_fixed_all
                )
            else:
                fixed_dist = None

            # Get particle
            particle = Particle(
                dist=dist,
                fixed_dist=fixed_dist,
                reward=0,
                pre_reward=0,
                info={},
                policy=initial_policy,
                memory=memory,
                optim=optim,
                id=p_ind,
                lifetime=0,
                twin=False,
                num_clone=0,
                just_spawned=False,
            )
            p_all += [particle]
        return p_all
