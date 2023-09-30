"""
Adaptation with particles.

"""
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
import torch
import shutil
import wandb
from omegaconf import OmegaConf

from adaptsim.agent.particle import ParticleFactory
from adaptsim.param.param_inference import ParamInference
from adaptsim.agent.util import util_agent_dict
from adaptsim.policy import policy_agent_dict
from adaptsim.param import param_agent_dict
from adaptsim.util.scheduler import StepLRFixed


class Adapt():

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
        self.use_wandb = cfg.wandb.entity is not None
        self.max_itr = cfg.adapt.max_itr
        self.num_adapt_step = cfg.adapt.num_adapt_step
        self.num_task_train = cfg.adapt.num_task_train
        self.num_meta_target_task = cfg.adapt.num_meta_target_task
        self.num_inner_target_task = cfg.adapt.num_inner_target_task
        self.num_meta_eval_episode = cfg.adapt.num_meta_eval_episode
        self.inference_update_freq = cfg.adapt.inference_update_freq
        self.batch_size = cfg.adapt.batch_size
        self.num_adapt_update = int(
            cfg.adapt.replay_ratio * self.inference_update_freq
            * self.num_adapt_step / self.batch_size
        )
        self.flag_eval_inner_target_reward = cfg.adapt.eval_target_reward
        self.pretrain_reward_check_freq = cfg.adapt.eval_freq
        self.reward_threshold_upper = cfg.adapt.reward_threshold_upper
        self.reward_threshold_lower = cfg.adapt.reward_threshold_lower

        # Pretraining related
        self.cfg_eps_init = cfg.adapt.eps_init
        self.cfg_eps = cfg.adapt.eps
        self.num_adapt_init = cfg.adapt.num_adapt_init
        self.num_adapt = cfg.adapt.num_adapt
        self.init_inference_update_wait = cfg.adapt.init_inference_update_wait

        # Spawn related
        self.num_particle_init = cfg.adapt.spawn.num_particle_init
        self.kill_freq = cfg.adapt.spawn.kill_freq
        self.inner_spawn_perturbation = cfg.adapt.spawn.inner_perturbation
        self.meta_spawn_perturbation = cfg.adapt.spawn.meta_perturbation
        self.spawn_reward_threshold = cfg.adapt.spawn.reward_threshold

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
        if hasattr(cfg.adapt, 'pretrain_policy_path'):
            self.pretrain_policy = cfg.pretrain_policy_path
        self.meta_policy_option = cfg.adapt.meta_policy_option
        self.inner_policy_option = cfg.adapt.inner_policy_option

        # Pre-trained optimizer
        self.pretrain_optim = None
        if hasattr(cfg.adapt, 'pretrain_optim_path'):
            self.pretrain_optim = torch.load(
                cfg.pretrain_optim_path, map_location=self.policy_agent.device
            )

        # Pre-trained memory
        self.pretrain_memory = None
        if hasattr(cfg.adapt, 'pretrain_memory_path'):
            self.pretrain_memory = self.policy_agent.make_meta_memory(
                cfg.pretrain_memory_path
            )

        # Initialize parameter agent
        logging.info('== Building parameter agent')
        self.param_agent = param_agent_dict[cfg.param.name](cfg.param)

        # Launch particles - assign pretrained policy and memory if exists
        logging.info('== Building initial particles')
        self.p_factory = ParticleFactory(
            cfg.seed,
            self.param_agent.adapt_param_cfg,
            self.param_agent.fixed_param_cfg,
        )
        self.p_all = self.p_factory.gen_space_filling_particle(
            self.num_particle_init,
            initial_policy=self.pretrain_policy,
            initial_memory=self.pretrain_memory,
            initial_optim=self.pretrain_optim,
            use_init_std=False,
        )
        # log particle in lines
        logging.info(
            'Particles: \n{}'.format(
                ''.join([str(p) + '\n' for p in self.p_all])
            )
        )

        # Generate meta target - uniform within distirbution
        self.meta_particle = self.p_factory.gen_particle(
            cfg.param_target.param,
            p_id=0,
        )
        logging.info('Target particle: {}'.format(str(self.meta_particle)))
        self.meta_target = self.param_agent.generate_task_wid(
            num_task=self.num_meta_target_task,
            particle=self.meta_particle,
            clip_param=False,
            uniform_goal=True,
        )
        logging.info('Target task: {}'.format(self.meta_target))

        # Initialize inference - borow some cfg from param_agent
        logging.info('== Building inference agent')
        cfg.inference.num_adapt_param = self.param_agent.num_adapt_param
        cfg.inference.num_param_bin = self.param_agent.num_param_bin
        self.infer_agent = ParamInference(cfg.inference)

        # Initialize utility agent - for helper functions specific to environment
        logging.info('== Building utility agent')
        self.util_agent = util_agent_dict[cfg.util.name](cfg.util)

    def run(self):
        # for convenience
        param_agent = self.param_agent
        util_agent = self.util_agent
        infer_agent = self.infer_agent
        policy_agent = self.policy_agent

        # Record histories of parameters for meta
        param_history = OmegaConf.create()
        for name in param_agent.adapt_param_name:
            param_history[name + '_lp'] = OmegaConf.create(
            )  # left param (mean or lower_bound)
            param_history[name + '_rp'] = OmegaConf.create(
            )  # right param (std or upper bound)

        # Run sim2sim or sim2real iterations
        for cnt_itr in range(self.max_itr):
            logging.info(f'New iteration: {cnt_itr}\n\n')

            ##################################################################
            ####################      Meta Level        ######################
            ##################################################################

            # Evaluate particles
            meta_eval_reward_all = []
            for p in self.p_all:
                # logging.info(f'Training particle {p.id}...')

                # Update tasks within param
                meta_train_tasks = param_agent.generate_task_wid(
                    self.num_task_train, p
                )

                # Train policy at meta level
                new_policy, _, new_optim = policy_agent.learn(
                    meta_train_tasks,
                    policy_path=p.policy,
                    optimizer_state=p.optim,
                    memory=p.memory,
                    out_folder_postfix='itr-{}_pid-{}_target'.format(
                        cnt_itr, p.id
                    ),
                    save_path=os.path.join(
                        self.fig_folder,
                        'itr-{}_pid-{}_target'.format(cnt_itr, p.id)
                    ),
                )

                # Evaluate on meta targets - using policy learned above
                meta_reward, meta_info = policy_agent.evaluate(
                    self.meta_target
                )

                # Evaluate, but not at pretrain
                if cnt_itr > 0:
                    logging.info(f'Evaluating particle {p.id}...')

                    # Evaluate - not for adaptation, just for checking performance - using policy from above
                    meta_eval_reward, _ = policy_agent.evaluate(
                        self.meta_target,
                        num_episode=self.num_meta_eval_episode
                    )
                    meta_eval_reward_all += [meta_eval_reward]

                    # Save to particle
                    p.reward = meta_reward
                    p.info = meta_info

                # Determine meta policy
                if self.meta_policy_option == 'raw':
                    p.policy = None
                    p.optim = None
                elif self.meta_policy_option == 'reuse':
                    p.policy = new_policy
                    p.optim = new_optim

                # Determine meta memory
                p.memory = None

            # Kill particles based on reward
            if self.kill_freq and cnt_itr % self.kill_freq == 0 and cnt_itr > 0:
                particle_value_all = [p.reward for p in self.p_all]
                flag_kill = len(self.p_all) > 1 and np.max(
                    particle_value_all
                ) > self.spawn_reward_threshold
                logging.info(
                    'Particle reward/value: {}'.format(particle_value_all)
                )
                if flag_kill:
                    worst_particle_index = np.argmin(particle_value_all)
                    best_particle_index = np.argmax(particle_value_all)
                    best_particle = self.p_all[best_particle_index]
                    worst_particle = self.p_all.pop(worst_particle_index)
                    logging.warning(
                        'Removed particle: {}!'.format(worst_particle)
                    )

                    # Spawn new ones - copy the best one
                    new_particle = self.p_factory.clone_particle(
                        best_particle,
                        perturbation=self.meta_spawn_perturbation
                    )
                    new_particle.id = worst_particle.id  # reuse id
                    self.p_all += [new_particle]
                    logging.warning(
                        'Spawned particle: {}!'.format(new_particle)
                    )

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

            ##################################################################
            ###################      Inner Level        ######################
            ##################################################################

            # Reset eps
            if cnt_itr == 0:
                cfg_eps = self.cfg_eps_init
            else:
                cfg_eps = self.cfg_eps
            eps_schduler = StepLRFixed(
                init_value=cfg_eps.init, period=cfg_eps.period,
                end_value=cfg_eps.end, step_size=cfg_eps.step
            )

            # Record
            loss = {}
            adapt_reward = {}
            adapt_cnt = 0
            reward_scaled_sum = []

            # Determine number of inner adaptations
            if cnt_itr == 0:
                num_adapt = self.num_adapt_init
            else:
                num_adapt = self.num_adapt
            with tqdm(total=num_adapt) as pbar:
                while adapt_cnt < num_adapt:

                    ########### Sample a cloned particle and target #########
                    p_ind = self.rng.choice(np.arange(len(self.p_all)))
                    p = self.p_all[p_ind]

                    # Clone
                    inner_p = self.p_factory.clone_particle(
                        p, perturbation=self.inner_spawn_perturbation
                    )

                    # Sample inner target using cloned particle
                    inner_target = param_agent.generate_task_ood(
                        num_task=self.num_inner_target_task, particle=inner_p
                    )

                    # Get optimal reward on target - only for dp_linearized
                    if self.flag_eval_inner_target_reward:
                        _, _, _ = policy_agent.learn(inner_target)
                        pre_reward, _ = policy_agent.evaluate(
                            inner_target, policy_path=inner_p.policy
                        )
                        inner_p.pre_reward = pre_reward

                    ################## Adapt #################

                    # Set policy back for evaluating pre_reward
                    if self.flag_eval_inner_target_reward:
                        inner_train_tasks = param_agent.generate_task_wid(
                            self.num_task_train, inner_p
                        )
                        new_policy, _, new_optim = policy_agent.learn(
                            tasks=inner_train_tasks
                        )

                    # Evaluate on inner target - using policy above
                    reward, info = policy_agent.evaluate(
                        inner_target, policy_path=inner_p.policy
                    )

                    # Get observation at target - reward, traj, etc - randomize order of tasks
                    target_obs = util_agent.get_target_obs(info)
                    target_latent = infer_agent.infer_traj(target_obs)

                    # Re-sample if pre-adapt reward is high
                    if reward > self.reward_threshold_upper:
                        continue

                    # Determine inner policy
                    if self.inner_policy_option == 'raw':
                        inner_p.policy = None
                        inner_p.optim = None

                    # Determine inner memory
                    inner_p.memory = None

                    # Inner adapt - like an episode
                    for step in range(self.num_adapt_step):

                        # Get state
                        state = np.hstack((
                            param_agent.get_state_representation(inner_p),
                            deepcopy(target_latent),
                        ))

                        # Infer
                        eps = eps_schduler.get_variable()
                        flag_random = self.rng.choice(2, p=[1 - eps, eps])
                        action = infer_agent.infer(
                            state,
                            random_flags=[flag_random],
                            verbose=False,
                        )[0]
                        param_agent.apply_action(action, inner_p)

                        # Sample new tasks within param
                        inner_train_tasks = param_agent.generate_task_wid(
                            self.num_task_train, inner_p
                        )

                        # Train on new param
                        new_policy, _, new_optim = policy_agent.learn(
                            inner_train_tasks,
                            policy_path=inner_p.policy,
                            optimizer_state=inner_p.optim,
                            memory=inner_p.memory,
                            out_folder_postfix='itr-{}_cnt-{}_step-{}_pid-{}'.
                            format(cnt_itr, adapt_cnt, step, inner_p.id),
                            save_path=os.path.join(
                                self.fig_folder,
                                'itr-{}_cnt-{}_step-{}_pid-{}'.format(
                                    cnt_itr, adapt_cnt, step, inner_p.id
                                )
                            ),
                        )

                        # Evaluate on target- use trained policy
                        reward, info = policy_agent.evaluate(inner_target)
                        target_obs = util_agent.get_target_obs(info)
                        target_latent = infer_agent.infer_traj(target_obs)

                        # Get info
                        reward_scaled = util_agent.scale_reward(
                            reward, pre_reward=inner_p.pre_reward
                        )
                        if reward_scaled < self.reward_threshold_lower:
                            reward_scaled = 0 * reward_scaled
                        next_state = np.hstack((
                            param_agent.get_state_representation(inner_p),
                            deepcopy(target_latent),
                        ))
                        done = False
                        if step == self.num_adapt_step - 1:
                            done = True

                        # Add data to buffer - use scaled reward - no need to save traj since not updating traj network
                        infer_agent.store_transition(
                            state, action[None], reward_scaled, next_state,
                            done, None
                        )

                        ########### Update for next step ###########

                        if self.inner_policy_option == 'raw':
                            inner_p.policy = None
                            inner_p.optim = None
                        elif self.inner_policy_option == 'reuse':
                            inner_p.policy = new_policy
                            inner_p.optim = new_optim

                        # Determine meta memory - use meta or use pre-trainnd
                        inner_p.memory = None

                        # Record reward progress
                        reward_scaled_sum += [reward_scaled]

                        # Quit if reward very high
                        if reward_scaled > self.reward_threshold_upper:
                            break

                    # Count
                    eps_schduler.step()
                    adapt_cnt += 1
                    pbar.update(1)

                    # Update inference
                    if adapt_cnt % self.inference_update_freq == 0 and not (
                        cnt_itr == 0
                        and adapt_cnt < self.init_inference_update_wait
                    ):
                        loss_batch = 0
                        for _ in range(self.num_adapt_update):
                            loss_batch += infer_agent.update_inference(
                                self.batch_size
                            )
                        logging.info(
                            'Update inference network for {} steps'.format(
                                self.num_adapt_update
                            )
                        )
                        loss[adapt_cnt] = loss_batch / self.num_adapt_update

                    # Plot for pretrain progress
                    if cnt_itr == 0 and adapt_cnt % self.pretrain_reward_check_freq == 0:
                        adapt_reward[adapt_cnt] = np.mean(reward_scaled_sum)
                        reward_scaled_sum = []
                        plt.plot(
                            list(adapt_reward.keys()),
                            list(adapt_reward.values())
                        )
                        plt.savefig(
                            os.path.join(
                                self.fig_folder,
                                str(cnt_itr) + '_reward.png'
                            )
                        )
                        plt.close()
                        plt.plot(list(loss.keys()), list(loss.values()))
                        plt.savefig(
                            os.path.join(
                                self.fig_folder,
                                str(cnt_itr) + '_loss.png'
                            )
                        )
                        plt.close()

                        # Save best inference model
                        past_adapt_reward = list(adapt_reward.values())[:-1]
                        if len(past_adapt_reward) > 0 and adapt_reward[
                            adapt_cnt] >= np.max(past_adapt_reward):
                            infer_agent.save(
                                cnt_itr, self.inference_model_folder
                            )
                            logging.info(
                                'Saved inference model with adapt reward {}'.
                                format(adapt_reward[adapt_cnt])
                            )

            ##################################################################
            ###################      Meta Level        ######################
            ##################################################################
            logging.info('Iteration done.\n')

            # Save inference network
            infer_agent.save(cnt_itr, self.inference_model_folder)

            # Record
            loss = np.mean(list(loss.values()))
            if self.use_wandb:
                wandb.log({"Inference loss": loss}, step=cnt_itr + 1,
                          commit=False)

            # Adapt particles
            if cnt_itr > 0:
                for p in self.p_all:

                    # Do not adapt if particle just spawned
                    if p.just_spawned:
                        p.just_spawned = False
                        continue

                    # Apply param action
                    meta_target_obs = util_agent.get_target_obs(p.info)
                    meta_target_latent = infer_agent.infer_traj(
                        meta_target_obs
                    )
                    state = np.hstack((
                        param_agent.get_state_representation(p),
                        deepcopy(meta_target_latent)
                    ))
                    action = infer_agent.infer(state, random_flags=[
                        False
                    ])[0]  # do not explore in real
                    param_agent.apply_action(action, p)
                    logging.info(
                        'Particle id {} updated param to {} with action {}.'.
                        format(p.id, p.dist, action)
                    )
