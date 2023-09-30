import os
import numpy as np
import wandb
import time
import logging
import torch
import shutil

from adaptsim.agent.particle import ParticleFactory
from adaptsim.param import infer_agent_dict
from adaptsim.param import param_agent_dict
from adaptsim.policy import policy_agent_dict
from adaptsim.util.plot import plot_posterior
from adaptsim.util.dist import Gaussian


class AdaptSysID():
    """
    System identification with point estimate or Bayessim.
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
        self.fig_folder = os.path.join(self.out_folder, 'fig')
        if os.path.exists(self.fig_folder):
            shutil.rmtree(self.fig_folder)
        os.makedirs(self.fig_folder)

        # Params
        self.use_wandb = cfg.use_wandb
        self.max_itr = cfg.max_itr
        self.num_task_train = cfg.num_task_train
        self.num_meta_target_task = cfg.num_meta_target_task
        self.num_meta_eval_target_task = cfg.num_meta_eval_target_task
        self.n_train_trajs = cfg.n_train_trajs
        self.traj_len = cfg.inference.traj_len
        self.flag_point_estimate = (cfg.inference.model.name == 'PointNN')
        self.num_traj_per_task = cfg.num_traj_per_task
        self.flag_finetune_trained_policy = cfg.flag_finetune_trained_policy
        self.traj_type = cfg.traj_type
        self.num_goal_per_task = cfg.num_goal_per_task if 'num_goal_per_task' in cfg else 1
        self.clip_param = cfg.clip_param

    def build_agents(self, cfg, venv):
        """
        Only one particle.
        """

        # Initialize policy
        logging.info('== Building policy agent')
        self.policy_agent = policy_agent_dict[cfg.policy.nam](cfg.policy, venv)

        # Initialize parameter agent
        logging.info('== Building param agent for adaptation')
        self.param_agent = param_agent_dict[cfg.param.name](cfg.param)

        # Launch particles - assign pretrained policy and memory if exists
        logging.info('== Building initial particles')
        self.p_factory = ParticleFactory(
            cfg.seed, self.param_agent.adapt_param_cfg,
            self.param_agent.fixed_param_cfg
        )
        self.p = self.p_factory.gen_space_filling_particle(
            1, initial_policy=None, initial_memory=None, initial_optim=None
        )[0]
        logging.info('Particles: {}'.format(self.p))
        self.p_orig = self.p_factory.clone_particle(self.p)

        # Generate meta target - uniform within distirbution
        self.meta_particle = self.p_factory.gen_particle(
            cfg.param_target.param, p_id=0
        )
        logging.info('Target particle: {}'.format(str(self.meta_particle)))
        self.meta_target = self.param_agent.generate_task_wid(
            num_task=self.num_meta_eval_target_task,
            particle=self.meta_particle,
            num_goal_per_task=self.num_goal_per_task, clip_param=False,
            uniform_goal=True
        )
        logging.info('Target task: {}'.format(self.meta_target))

        # Initialize inference - borow some cfg from param_agent
        logging.info('== Building inference agent')
        cfg.inference.model.output_lows = self.param_agent.adapt_param_lower_bound.tolist(
        )  # adapt, then fixed
        cfg.inference.model.output_highs = self.param_agent.adapt_param_upper_bound.tolist(
        )
        cfg.inference.model.output_dim = self.param_agent.num_adapt_param
        self.infer_agent = infer_agent_dict[cfg.inference.name](cfg.inference)
        self.inference_model_path = cfg.inference.model_path

    def run(self):

        # for convenience
        param_agent = self.param_agent
        infer_agent = self.infer_agent
        policy_agent = self.policy_agent
        p = self.p
        p_orig = self.p_orig
        trained_policy = None

        # Run sim2sim or sim2real iterations
        for cnt_itr in range(self.max_itr):
            logging.info(f'New iteration: {cnt_itr}\n\n')
            t = time.time()

            ##################################################################
            ####################      Meta Level        ######################
            ##################################################################

            # Plot
            figs = plot_posterior(
                sim_params_names=param_agent.adapt_param_name,
                skip_ids=[],
                true_params=self.meta_particle.dist.lb_array,  # lb=ub
                posterior=p.dist,
                p_lower=param_agent.adapt_param_lower_bound,
                p_upper=param_agent.adapt_param_upper_bound,
                output_file=os.path.join(self.fig_folder, str(cnt_itr))
            )

            # Set best policy for current distribution
            meta_train_tasks = param_agent.generate_task_wid(
                self.num_task_train, p, clip_param=self.clip_param
            )
            meta_policy, _, _ = policy_agent.learn(
                meta_train_tasks, policy_path=trained_policy,
                save_path=os.path.join(self.fig_folder, str(cnt_itr)),
                out_folder_postfix='itr-{}-meta'.format(cnt_itr)
            )

            # Evaluate on meta params - once for each meta task
            meta_reward, _ = policy_agent.evaluate(
                tasks=self.meta_target, policy_path=meta_policy
            )

            # If finetune, save the trained policy
            if self.flag_finetune_trained_policy:
                trained_policy = meta_policy

            # Report
            logging.info(f'= Iteration {cnt_itr} with meta policy updated')
            logging.info(f'= Meta reward: {meta_reward}')
            if self.use_wandb:
                wandb.log({'Best Meta Reward': meta_reward}, step=cnt_itr,
                          commit=True)

            ##################################################################
            ###################      Inner Level        ######################
            ##################################################################

            # Collect trajectories for BayesSim training and train BayesSim.
            # In their implementation, this distribution is fixed for all iterations, which means the inference model is trained with full parameter range always, the difference is the data collection policy, which is iteratively trained on environments closer to the target.
            infer_agent.reset_model(
                path=self.inference_model_path
            )  # or fine-tune

            # Set to full range for data collection
            inner_tasks = param_agent.generate_task_wid(
                self.num_task_train, p_orig,
                num_goal_per_task=self.num_goal_per_task,
                clip_param=self.clip_param
            )

            # Use newly trained policy to collect data
            self.policy_agent.reset_policy(meta_policy)

            n_trajs_done = 0
            logging.info(f'Will train BayesSim on {self.n_train_trajs} trajs')
            while n_trajs_done < self.n_train_trajs:
                n_trajs_per_batch = infer_agent.get_n_trajs_per_batch(
                    self.n_train_trajs, n_trajs_done
                )

                # Sample with replacement - acccount for goal per task
                num_task_param_train = int(
                    self.num_task_train / self.num_goal_per_task
                )
                num_task_batch = int(
                    n_trajs_per_batch / self.num_goal_per_task
                )
                batch_tasks = [
                    inner_tasks[ind * self.num_goal_per_task + incre]
                    for ind in
                    self.rng.choice(num_task_param_train, num_task_batch)
                    for incre in range(self.num_goal_per_task)
                ]

                # use inner policy, add perturbation
                # logging.info(f'Collect {n_trajs_per_batch} trajs')
                sim_params, sim_states, sim_actions, _ = self.policy_agent.collect_data(
                    batch_tasks, perturb_policy=True,
                    num_traj_per_task=self.num_traj_per_task,
                    max_traj_len=self.traj_len, traj_type=self.traj_type
                )
                num_episode, num_traj_per_task, _, _ = sim_states.shape  # we check num_traj_per_task again since in dp, we filter out some trajs of the task
                num_task_batch = int(
                    num_episode / self.num_goal_per_task
                )  # just for dp filtering unstable trajectories, assuming num_goal_per_task=1

                # num_episode x num_traj_per_task x max_traj_len x obs_dim -> num_task_batch x num_goal_per_task x num_traj_per_task x max_traj_len x obs_dim
                sim_states = sim_states.reshape(
                    num_task_batch, self.num_goal_per_task, num_traj_per_task,
                    self.traj_len, -1
                )
                sim_actions = sim_actions.reshape(
                    num_task_batch, self.num_goal_per_task, num_traj_per_task,
                    self.traj_len, -1
                )

                # Index sim_params every num_goal_per_task -> num_task_batch x param_dim
                sim_params = sim_params[::self.num_goal_per_task]

                # Train
                logging.info('Train BayesSim...')
                log_bsim = infer_agent.run_training(
                    sim_params, sim_states, sim_actions
                )
                n_trajs_done += n_trajs_per_batch
                logging.info(
                    f'{n_trajs_done:d} of {self.n_train_trajs:d} done'
                )

            ##########################################################
            # Update posterior using surrogate real trajs and BayesSim inference.
            ##########################################################

            # Generate random targets
            new_real_target = self.param_agent.generate_task_wid(
                num_task=self.num_meta_target_task,
                particle=self.meta_particle,
                num_goal_per_task=self.num_goal_per_task,
                clip_param=self.clip_param, uniform_goal=True
            )

            # Collect data on meta params, using meta policy, add perturbation
            self.policy_agent.reset_policy(meta_policy)
            _, real_states, real_actions, *_ = self.policy_agent.collect_data(
                new_real_target, num_traj_per_task=self.num_traj_per_task,
                max_traj_len=self.traj_len, traj_type=self.traj_type
            )
            num_episode, num_traj_per_task, _, _ = real_states.shape  # we check num_traj_per_task again since in dp, we filter out some trajs of the task
            num_task_batch = int(
                num_episode / self.num_goal_per_task
            )  # just for dp filtering unstable trajectories, assuming num_goal_per_task=1

            # num_episode x num_traj_per_task x max_traj_len x obs_dim -> num_task_batch x num_goal_per_task x num_traj_per_task x max_traj_len x obs_dim
            # num_task_batch = int(self.num_meta_target_task / self.num_goal_per_task)
            real_states = real_states.reshape(
                num_task_batch, self.num_goal_per_task, num_traj_per_task,
                self.traj_len, -1
            )
            real_actions = real_actions.reshape(
                num_task_batch, self.num_goal_per_task, num_traj_per_task,
                self.traj_len, -1
            )

            # Concatenate and infer
            if cnt_itr == 0 or self.flag_point_estimate:
                all_real_states = real_states
                all_real_actions = real_actions
            else:
                all_real_states = torch.cat([all_real_states, real_states],
                                            dim=0)
                all_real_actions = torch.cat([all_real_actions, real_actions],
                                             dim=0)
            sim_params_distr = infer_agent.predict(
                all_real_states, all_real_actions
            )  # num_task x num_goal_per_task x num_traj_per_task x max_len x state/action_dim

            # Make a very thin Gaussian for point estimate
            if self.flag_point_estimate:
                sim_params_distr = Gaussian(
                    m=sim_params_distr,
                    L=np.ones((len(sim_params_distr))) * 1e-6
                )

            p.dist = sim_params_distr
            logging.info('Updated particle to {}!'.format(p.dist))

            # Done with iteration
            logging.info(f'Time for itr {cnt_itr}: {time.time()-t}\n')

            # Save model
            if cnt_itr > 0:
                infer_agent.save_model(
                    os.path.join(self.inference_model_folder, str(cnt_itr))
                )
