import os
import numpy as np
import wandb
import time
import torch
from copy import deepcopy
import logging
from omegaconf import OmegaConf
import shutil

from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf

from adaptsim.agent.particle import ParticleFactory
from adaptsim.policy import policy_agent_dict
from adaptsim.param import param_agent_dict
from adaptsim.util.numeric import standardize, unnormalize


class AdaptBayesOpt():
    """
    Class for adaptation with particles.
    """

    def __init__(self, cfg, venv):
        self.store_cfg(cfg)
        self.build_agents(cfg, venv)

    def store_cfg(self, cfg):
        self.rng = np.random.default_rng(seed=cfg.seed)

        # Output
        self.out_folder = cfg.out_folder
        self.inference_model_folder = os.path.join(self.out_folder, 'model')
        os.makedirs(self.inference_model_folder, exist_ok=True)
        self.fig_folder = os.path.join(self.out_folder, 'fig')
        if os.path.exists(self.fig_folder):
            shutil.rmtree(self.fig_folder)
        os.makedirs(self.fig_folder)

        # Params
        self.use_wandb = cfg.use_wandb
        self.max_itr = cfg.max_itr
        self.num_task_train = cfg.num_task_train
        self.num_meta_target_task = cfg.num_meta_target_task
        self.num_meta_eval_episode = cfg.num_meta_eval_episode
        self.num_meta_test_episode = cfg.num_meta_test_episode

        # Bayesian Optimization
        self.num_particle_init = cfg.spawn.num_particle_init
        self.acquisition_function_type = cfg.acquisition_function_type
        self.acquisition_restarts = cfg.acquisition_restarts
        self.acquisition_samples = cfg.acquisition_samples

    def build_agents(self, cfg, venv):
        """
        Multiple particles - sample distributions.
        """

        # Initialize policy
        logging.info('== Building policy agent')
        self.policy_agent = policy_agent_dict[cfg.policy.name
                                             ](cfg.policy, venv)

        # Pre-trained policy
        self.pretrain_policy = None
        if hasattr(cfg, 'pretrain_policy_path'):
            self.pretrain_policy = cfg.pretrain_policy_path
        self.meta_policy_option = cfg.meta_policy_option
        self.inner_policy_option = cfg.inner_policy_option

        # Pre-trained optimizer
        self.pretrain_optim = None
        if hasattr(cfg, 'pretrain_optim_path'):
            self.pretrain_optim = torch.load(
                cfg.pretrain_optim_path, map_location=self.policy_agent.device
            )

        # Pre-trained memory
        self.pretrain_memory = None
        if hasattr(cfg, 'pretrain_memory_path'):
            self.pretrain_memory = self.policy_agent.make_meta_memory(
                cfg.pretrain_memory_path
            )

        # Initialize parameter agent
        logging.info('== Building param agent for adaptation')
        self.param_agent = param_agent_dict[cfg.param.name](cfg.param)

        # Launch particles - assign pretrained policy and memory if exists
        logging.info('== Building initial particles')
        self.p_factory = ParticleFactory(
            cfg.seed, self.param_agent.adapt_param_cfg,
            self.param_agent.fixed_param_cfg
        )
        self.p_all = self.p_factory.gen_space_filling_particle(
            self.num_particle_init, initial_policy=self.pretrain_policy,
            initial_memory=self.pretrain_memory,
            initial_optim=self.pretrain_optim, use_init_std=False
        )  # gaussian around init_std
        logging.info('Particles: {}'.format([str(p) for p in self.p_all]))

        # Generate meta target (OOD) - uniform within distirbution
        self.meta_particle = self.p_factory.gen_particle(
            cfg.param_target.param, p_id=0
        )
        logging.info('Target particle: {}'.format(str(self.meta_particle)))
        self.meta_target = self.param_agent.generate_task_wid(
            num_task=self.num_meta_target_task, particle=self.meta_particle,
            clip_param=False, uniform_goal=True
        )
        logging.info('Target task: {}'.format(self.meta_target))

    def run(self):
        # for convenience
        param_agent = self.param_agent
        policy_agent = self.policy_agent

        # Record histories of parameters for meta
        param_history = OmegaConf.create()
        for name in param_agent.adapt_param_name:
            param_history[name + '_lp'] = OmegaConf.create(
            )  # left param (mean or lower_bound)
            param_history[name + '_rp'] = OmegaConf.create(
            )  # right param (std or upper bound)

        # Run sim2sim or sim2real iterations
        meta_eval_reward_all = []
        meta_reward_all = []
        for cnt_itr in range(self.max_itr):
            logging.info(f'New iteration: {cnt_itr}\n\n')
            t = time.time()

            ####################      Meta Level     ######################

            # Evaluate particles
            for p_ind, p in enumerate(self.p_all):
                logging.info(f'Training particle {p.id}...')

                # Train policy at meta level - only for new ones
                if cnt_itr == 0 or p_ind == len(self.p_all) - 1:
                    # Update tasks within param
                    meta_train_tasks = param_agent.generate_task_wid(
                        self.num_task_train, p
                    )

                    new_policy, new_memory, new_optim = policy_agent.learn(
                        meta_train_tasks, policy_path=p.policy,
                        optimizer_state=p.optim, memory=p.memory,
                        out_folder_postfix='itr-{}_pid-{}_target'.format(
                            cnt_itr, p.id
                        ), save_path=os.path.join(
                            self.fig_folder,
                            'itr-{}_pid-{}_target'.format(cnt_itr, p.id)
                        )
                    )

                    # Determine meta policy
                    if self.meta_policy_option == 'raw':
                        p.policy = None
                        p.optim = None
                    elif self.meta_policy_option == 'reuse':
                        p.policy = new_policy
                        p.optim = new_optim

                    # Determine meta memory
                    p.memory = None

                    # Evaluate on meta targets - using policy learned above
                    meta_reward, meta_info = policy_agent.evaluate(
                        self.meta_target, policy_path=p.policy,
                        num_episode=self.num_meta_test_episode
                    )
                    meta_reward_all += [meta_reward]

                    # Evaluate, but not at pretrain
                    logging.info(f'Evaluating particle {p.id}...')

                    # Evaluate - not for adaptation, just for checking performance - using policy from above
                    meta_eval_reward, _ = policy_agent.evaluate(
                        self.meta_target,
                        policy_path=p.policy,
                        num_episode=self.num_meta_eval_episode,
                    )
                    meta_eval_reward_all += [meta_eval_reward]
                    logging.info(f'Eval reward: {meta_eval_reward}')

                    # Save to particle
                    p.reward = meta_reward  # actually average
                    p.info = meta_info

            # Report
            if cnt_itr > 0:
                best_meta_reward = np.max(meta_eval_reward_all)
                logging.info(f'= Iteration {cnt_itr} with meta policy updated')
                logging.info(f'= Best meta reward: {best_meta_reward}')
                out = {}
                if self.use_wandb:
                    for ind, p in enumerate(self.p_all):
                        out["Meta reward for p_{}".format(p.id)
                           ] = meta_eval_reward_all[ind]
                    out["Best Meta Reward"] = best_meta_reward
                    wandb.log(out, step=cnt_itr, commit=True)

            ##############      Bayesian Optimization      ##############

            # Normalize the input data and standardize the output data
            x_bayesopt = []
            for p in self.p_all:
                x_bayesopt.append(param_agent.get_state_representation(p))
            x_bayesopt = torch.from_numpy(np.vstack(x_bayesopt)).double()
            y_bayesopt = torch.tensor(meta_reward_all).double()
            y_bayesopt = standardize(y_bayesopt).unsqueeze(1)
            print(x_bayesopt, y_bayesopt)

            # Create and fit the GP model
            gp = SingleTaskGP(x_bayesopt, y_bayesopt)
            gp.likelihood.noise_covar.register_constraint(
                "raw_noise", GreaterThan(1e-5)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
            logging.info("Fitted the GP.")

            # Acquisition functions
            if self.acquisition_function_type == "UCB":
                acq_fcn = UpperConfidenceBound(
                    gp, beta=self.acq_param.get("beta", 0.1), maximize=True
                )
            elif self.acquisition_function_type == "EI":  # used in the paper
                acq_fcn = ExpectedImprovement(
                    gp, best_f=y_bayesopt.max().item(), maximize=True
                )
            elif self.acquisition_function_type == "PI":
                acq_fcn = ProbabilityOfImprovement(
                    gp,
                    best_f=y_bayesopt.max().item(),
                    maximize=True,
                )
            else:
                raise 'Unknown acquisition function type!'

            # Optimize acquisition function and get new candidate point
            next_x, _ = optimize_acqf(
                acq_function=acq_fcn,
                bounds=torch.stack([
                    torch.zeros(param_agent.num_adapt_param),
                    torch.ones(param_agent.num_adapt_param)
                ]).to(dtype=torch.float32),
                q=1,
                num_restarts=self.acquisition_restarts,
                raw_samples=self.acquisition_samples,
            )
            next_x = next_x.to(dtype=torch.get_default_dtype()).numpy()[0]
            logging.info(f"Found the next candidate: {next_x}")

            # Add new candicate to p_all
            new_p = deepcopy(self.p_all[0])
            new_p.id = len(self.p_all)
            new_p.dist.mean[:param_agent.num_adapt_param] = unnormalize(
                next_x, param_agent.adapt_param_lower_bound,
                param_agent.adapt_param_upper_bound
            )
            new_p.policy = None
            self.p_all.append(new_p)

            # Done with iteration
            logging.info(f'Time for itr {cnt_itr}: {time.time()-t}\n')
