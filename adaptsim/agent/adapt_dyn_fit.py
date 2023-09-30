import os
import numpy as np
import wandb
import logging

from adaptsim.agent.particle import ParticleFactory
from adaptsim.policy import policy_agent_dict
from adaptsim.param import param_agent_dict


class AdaptDynFit():
    """
    Fitting dynamics directly in target environment. For the pushing task, map actions to final position, then infer policy as nearest neighbor.
    """

    def __init__(self, cfg, venv):
        self.store_cfg(cfg)
        self.build_agents(cfg, venv)

    def store_cfg(self, cfg):

        # RNG
        self.rng = np.random.default_rng(seed=cfg.seed)

        # Output
        self.out_folder = cfg.out_folder
        self.inference_model_folder = os.path.join(self.out_folder, 'model')
        os.makedirs(self.inference_model_folder, exist_ok=True)

        # Params
        self.use_wandb = cfg.use_wandb
        self.num_meta_target_task = cfg.num_meta_target_task
        self.num_meta_eval_target_task = cfg.num_meta_eval_target_task
        self.num_traj_per_task = cfg.num_traj_per_task
        self.traj_type = cfg.traj_type

    def build_agents(self, cfg, venv):

        # Initialize policy
        logging.info('== Building policy agent')
        self.policy_agent = policy_agent_dict[cfg.policy.name
                                             ](cfg.policy, venv)

        # Initialize parameter agent
        logging.info('== Building param agent for adaptation')
        self.param_agent = param_agent_dict[cfg.param.name](cfg.param)

        # Launch particles - assign pretrained policy and memory if exists
        logging.info('== Building initial particles')
        self.p_factory = ParticleFactory(
            cfg.seed, self.param_agent.adapt_param_cfg,
            self.param_agent.fixed_param_cfg
        )

        # Generate meta target - uniform within distirbution
        self.meta_particle = self.p_factory.gen_particle(
            cfg.param_target.param, p_id=0
        )
        logging.info('Target particle: {}'.format(str(self.meta_particle)))
        self.meta_target = self.param_agent.generate_task_wid(
            num_task=self.num_meta_eval_target_task,
            particle=self.meta_particle, clip_param=False, uniform_goal=True
        )
        logging.info('Target task: {}'.format(self.meta_target))

    def run(self):
        cnt_itr = 0

        # Generate target tasks
        target_tasks = self.param_agent.generate_task_wid(
            num_task=self.num_meta_target_task,
            particle=self.meta_particle,
            clip_param=False,
            uniform_goal=True,
        )

        # Collect trajectory at target - raw actions
        _, target_states, target_actions, *_ = self.policy_agent.collect_data(
            target_tasks,
            force_random=True,  # use random action
            num_traj_per_task=self.num_traj_per_task,
            traj_type=self.traj_type,
        )

        # Fit dynamics
        self.policy_agent.learn(target_states, target_actions)

        # Test policy
        meta_reward, _ = self.policy_agent.evaluate(self.meta_target)
        logging.info(f'= Meta reward: {meta_reward}')
        if self.use_wandb:
            wandb.log({'Best Meta Reward': meta_reward}, step=cnt_itr,
                      commit=True)