"""
Adaptation with particles, with multiple trainers.

"""
import os
import numpy as np
import wandb
import torch
from copy import deepcopy
from tqdm import tqdm
import logging
import shutil
import pickle
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from adaptsim.agent.particle_env import ParticleEnv, VecEnvParticle, assign_val
from adaptsim.agent.particle import ParticleFactory, find_twin_batch
from adaptsim.param.param_inference import ParamInference
from adaptsim.agent.util import util_agent_dict
from adaptsim.param import param_agent_dict
from adaptsim.util.scheduler import StepLRFixed


class AdaptTrainer():

    def __init__(self, cfg, venv_all):
        self.store_cfg(cfg)
        self.build_agents(cfg, venv_all)

    def store_cfg(self, cfg):
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.num_trainer = cfg.num_trainer
        self.num_env_per_trainer = cfg.num_env_per_trainer

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
        self.num_meta_test_task = cfg.adapt.num_meta_test_task
        self.inference_update_freq = cfg.adapt.inference_update_freq
        self.batch_size = cfg.adapt.batch_size
        self.num_adapt_update = int(
            cfg.adapt.replay_ratio * self.inference_update_freq
            * self.num_adapt_step / self.batch_size
        )
        self.pretrain_reward_check_freq = cfg.adapt.eval_freq
        self.twin_threshold = cfg.adapt.twin_threshold
        self.lifetime_threshold = cfg.adapt.lifetime_threshold
        self.reward_threshold = cfg.adapt.reward_threshold

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

    def build_agents(self, cfg, venv_all):
        """
        Multiple particles - sample distributions.
        """

        # Initialize particle env - spawn policy agent there
        trainers = [
            ParticleEnv(cfg.policy, venv_all[rank], seed=cfg.seed + rank)
            for rank in range(self.num_trainer)
        ]
        self.trainers = VecEnvParticle(
            trainers,
            pickle_option='cloudpickle',
            start_method='fork',
            fig_folder=self.fig_folder,
        )

        # Pre-trained policy
        self.pretrain_policy = None
        if hasattr(cfg, 'pretrain_policy_path'):
            self.pretrain_policy = cfg.adapt.pretrain_policy_path
        self.meta_policy_option = cfg.adapt.meta_policy_option
        self.inner_policy_option = cfg.adapt.inner_policy_option

        # Pre-trained optimizer
        self.pretrain_optim = None
        if hasattr(cfg, 'pretrain_optim_path'):
            self.pretrain_optim = torch.load(
                cfg.adapt.pretrain_optim_path,
                map_location=self.policy_agent.device
            )

        # Pre-trained memory
        self.pretrain_memory = None
        if hasattr(cfg, 'pretrain_memory_path'):
            self.pretrain_memory = self.policy_agent.make_meta_memory(
                cfg.adapt.pretrain_memory_path
            )

        # Initialize parameter agent
        logging.info('== Building param agent for adaptation')
        self.param_agent = param_agent_dict[cfg.param.name](cfg.param)

        # Launch particles - assign pretrained policy and memory if exists
        logging.info('== Setting up particles')
        self.p_factory = ParticleFactory(
            cfg.seed,
            self.param_agent.adapt_param_cfg,
            self.param_agent.fixed_param_cfg,
        )
        if not hasattr(cfg, 'particle_path'):
            self.p_init = self.p_factory.gen_space_filling_particle(
                self.num_particle_init,
                initial_policy=self.pretrain_policy,
                initial_memory=None,
                initial_optim=None,
                use_init_std=False,
            )  # gaussian around init_std
            self.p_all = deepcopy(self.p_init)
            logging.info(
                'Spawned {} init particles!'.format(self.num_particle_init)
            )
        else:
            with open(cfg.particle_path, 'rb') as f:
                data = pickle.load(f)
            p_init = data['init_particles']
            self.p_all = data['particles']
            logging.info(
                'Loaded init and all particles from {}!'.format(
                    cfg.particle_path
                )
            )

            # Sample
            if self.num_particle_init == 1:  # use particle closest to the mean if only using one; use normalized distance
                min_normalized_dist_to_mean = 1
                for p in p_init:
                    normalized_val = (
                        p.dist.mean - self.param_agent.adapt_param_lower_bound
                    ) / (
                        self.param_agent.adapt_param_upper_bound
                        - self.param_agent.adapt_param_lower_bound
                    )
                    normalized_dist_to_mean = np.linalg.norm(
                        normalized_val - 0.5 * np.ones_like(normalized_val)
                    )
                    if normalized_dist_to_mean < min_normalized_dist_to_mean:
                        min_normalized_dist_to_mean = normalized_dist_to_mean
                        self.p_init = [p]
            else:
                p_ind_batch = self.rng.choice(
                    np.arange(len(p_init)), size=self.num_particle_init,
                    replace=False
                )
                self.p_init = [p_init[ind] for ind in p_ind_batch]

        logging.info(
            'Particles: \n{}'.format(
                ''.join([str(p) + '\n' for p in self.p_all])
            )
        )
        self.p_id_cnt = len(self.p_all)

        # Generate meta target - uniform within distirbution
        self.meta_particle = self.p_factory.gen_particle(
            cfg.param_target.param, p_id=0
        )
        # logging.info('Target particle: {}'.format(str(self.meta_particle)))
        self.meta_target = self.param_agent.generate_task_wid(
            num_task=self.num_meta_target_task,
            particle=self.meta_particle,
            clip_param=False,
            uniform_goal=True,
        )
        # logging.info('Target task: {}'.format(self.meta_target))
        self.meta_test_target = self.param_agent.generate_task_wid(
            num_task=self.num_meta_test_task,
            particle=self.meta_particle,
            clip_param=False,
            uniform_goal=True,
        )
        # logging.info('Target test task: {}'.format(self.meta_test_target))

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

            # Evaluate particles in batches
            meta_eval_reward_all = []
            p_batch_all = [
                self.p_init[i:i + self.num_trainer]
                for i in range(0, len(self.p_init), self.num_trainer)
            ]
            for p_batch in p_batch_all:
                p_id_batch = [p.id for p in p_batch]
                logging.info(f'Training particle {p_id_batch}...')

                # Check if old particle is close enough - if so, re-use particle
                twin_batch, flag_train_batch = find_twin_batch(
                    p_batch, self.p_all, self.twin_threshold,
                    param_agent.adapt_param_range, self.lifetime_threshold
                )

                # Mark twin - sample tasks
                meta_train_tasks_batch = []
                for p, twin, flag_train in zip(
                    p_batch, twin_batch, flag_train_batch
                ):
                    if twin is not None:
                        p.lifetime = twin.lifetime
                    p.twin = False
                    if flag_train:
                        p.lifetime += 1
                        meta_train_tasks_batch += [
                            param_agent.generate_task_wid(
                                self.num_task_train, p
                            )
                        ]
                    else:
                        meta_train_tasks_batch += [[]]
                        if twin is not None:
                            p.twin = True

                # Train if needed
                self.trainers.reset(p_batch)
                new_policy_batch, _, _ = self.trainers.learn(
                    meta_train_tasks_batch, cnt_itr=cnt_itr,
                    pid_all=p_id_batch, twin_batch=twin_batch
                )

                # Determine meta policy
                if self.meta_policy_option == 'raw':
                    assign_val(p_batch, [('policy', None)])
                elif self.meta_policy_option == 'reuse':
                    assign_val(p_batch, [('policy', new_policy_batch)])

                # Evaluate, but not at pretrain
                if cnt_itr > 0:
                    logging.info(
                        f'Evaluating particle {p_id_batch} on meta target...'
                    )

                    # Evaluate - not for adaptation, just for checking performance - using policy from above
                    self.trainers.reset(p_batch)
                    meta_reward_batch, meta_info_batch = self.trainers.evaluate(
                        self.meta_target
                    )

                    # Save to particle
                    assign_val(
                        p_batch, [('reward', meta_reward_batch),
                                  ('info', meta_info_batch)]
                    )

                    # Test
                    logging.info(
                        f'Evaluating particle {p_id_batch} on meta test target...'
                    )
                    self.trainers.reset(p_batch)
                    meta_eval_reward_batch, _ = self.trainers.evaluate(
                        self.meta_test_target
                    )
                    meta_eval_reward_all += meta_eval_reward_batch

            # Kill particles based on V(S) or reward
            if self.kill_freq and cnt_itr % self.kill_freq == 0 and cnt_itr > 0:
                particle_value_all = [p.reward for p in self.p_init]
                flag_kill = len(self.p_init) > 1 and np.max(
                    particle_value_all
                ) > self.spawn_reward_threshold
                logging.info('Particle values: {}'.format(particle_value_all))

                if flag_kill:
                    worst_particle_index = np.argmin(particle_value_all)
                    best_particle_index = np.argmax(particle_value_all)
                    best_particle = self.p_init[best_particle_index]
                    worst_particle = self.p_init.pop(worst_particle_index)
                    logging.warning(
                        'Removed particle: {}!'.format(worst_particle)
                    )

                    # Spawn new ones - copy the best one
                    new_particle = self.p_factory.clone_particle(
                        best_particle,
                        perturbation=self.meta_spawn_perturbation
                    )
                    new_particle.id = worst_particle.id  # reuse id
                    new_particle.just_spawned = True
                    self.p_init += [new_particle]
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
                    for ind, p in enumerate(self.p_init):
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

                    ########### Sample cloned particles and target #########
                    p_ind_batch = self.rng.choice(
                        np.arange(len(self.p_init)), size=self.num_trainer,
                        replace=False
                    )
                    p_old_batch = [self.p_init[ind] for ind in p_ind_batch]

                    # Clone - assign a new id, perturb std - set twin=True
                    p_batch = []
                    for p_old in p_old_batch:
                        p_new = self.p_factory.clone_particle(
                            p_old, perturbation=self.inner_spawn_perturbation,
                            p_id=self.p_id_cnt
                        )
                        self.p_id_cnt += 1
                        p_batch += [p_new]
                        self.p_all += [deepcopy(p_new)]

                    # Sample inner target using cloned particle
                    inner_target_batch = [
                        param_agent.generate_task_ood(
                            num_task=self.num_inner_target_task, particle=p
                        ) for p in p_batch
                    ]

                    ################## Adapt #################

                    # Check if old particle is close enough - if so, re-use particle
                    twin_batch, flag_train_batch = find_twin_batch(
                        p_batch,
                        self.p_all,
                        self.twin_threshold,
                        param_agent.adapt_param_range,
                        self.lifetime_threshold,
                    )

                    # Mark twin - sample tasks
                    inner_train_tasks_batch = []
                    for p, twin, flag_train in zip(
                        p_batch, twin_batch, flag_train_batch
                    ):
                        if twin is not None:
                            p.lifetime = twin.lifetime
                        p.twin = False
                        if flag_train:
                            p.lifetime += 1
                            inner_train_tasks_batch += [
                                param_agent.generate_task_wid(
                                    self.num_task_train, p
                                )
                            ]
                        else:
                            inner_train_tasks_batch += [[]]
                            if twin is not None:
                                p.twin = True

                    # Train on new param
                    self.trainers.reset(p_batch)
                    new_policy_batch, _, _ = self.trainers.learn(
                        inner_train_tasks_batch, cnt_itr=cnt_itr,
                        pid_all=[p.id for p in p_batch],
                        adapt_cnt_base=adapt_cnt, step=-1,
                        twin_batch=twin_batch
                    )

                    # Update policy/memory/optim for particle
                    if self.inner_policy_option == 'raw':
                        assign_val(p_batch, [('policy', None)])
                    elif self.inner_policy_option == 'reuse':
                        assign_val(p_batch, [('policy', new_policy_batch)])

                    # Evaluate on inner target
                    self.trainers.reset(p_batch)
                    reward_batch, info_batch = self.trainers.evaluate(
                        inner_target_batch
                    )

                    # Get observation at target - reward, traj, etc - randomize order of tasks
                    target_obs_batch = np.vstack([
                        util_agent.get_target_obs(info) for info in info_batch
                    ])
                    target_latent_batch = infer_agent.infer_traj(
                        target_obs_batch
                    )

                    # Determine inner policy
                    if self.inner_policy_option == 'raw':
                        assign_val(p_batch, [('policy', None)])

                    # Inner adapt - like an episode
                    for step in range(self.num_adapt_step):

                        # Get state
                        state_batch = np.vstack([
                            param_agent.get_state_representation(p)
                            for p in p_batch
                        ])
                        state_batch = np.hstack((
                            state_batch,
                            deepcopy(target_latent_batch),
                        ))

                        # Infer
                        eps = eps_schduler.get_variable()
                        flags_random = self.rng.choice(
                            2, p=[1 - eps, eps], size=(self.num_trainer, 1)
                        )
                        action_batch = infer_agent.infer(
                            state_batch, random_flags=flags_random
                        )
                        for (action, p) in zip(action_batch, p_batch):
                            param_agent.apply_action(action, p)

                        # Check if old particle is close enough - if so, re-use particle
                        twin_batch, flag_train_batch = find_twin_batch(
                            p_batch, self.p_all, self.twin_threshold,
                            param_agent.adapt_param_range,
                            self.lifetime_threshold
                        )

                        # Mark twin - sample tasks
                        inner_train_tasks_batch = []
                        for p, twin, flag_train in zip(
                            p_batch, twin_batch, flag_train_batch
                        ):
                            if twin is not None:
                                p.lifetime = twin.lifetime
                            p.twin = False
                            if flag_train:
                                p.lifetime += 1
                                inner_train_tasks_batch += [
                                    param_agent.generate_task_wid(
                                        self.num_task_train, p
                                    )
                                ]
                            else:
                                inner_train_tasks_batch += [[]]
                                if twin is not None:
                                    p.twin = True

                        # Train on new param
                        self.trainers.reset(p_batch)
                        new_policy_batch, _, _ = self.trainers.learn(
                            inner_train_tasks_batch,
                            cnt_itr=cnt_itr,
                            pid_all=[p.id for p in p_batch],
                            adapt_cnt_base=adapt_cnt,
                            step=step,
                            twin_batch=twin_batch,
                        )

                        # Update policy for particle
                        if self.inner_policy_option == 'raw':
                            assign_val(p_batch, [('policy', None)])
                        elif self.inner_policy_option == 'reuse':
                            assign_val(p_batch, [('policy', new_policy_batch)])

                        # Evaluate on target- use trained policy
                        self.trainers.reset(p_batch)
                        reward_batch, info_batch = self.trainers.evaluate(
                            inner_target_batch
                        )
                        target_obs_batch = np.vstack([
                            util_agent.get_target_obs(info)
                            for info in info_batch
                        ])
                        target_latent_batch = infer_agent.infer_traj(
                            target_obs_batch
                        )

                        # Get info
                        reward_scaled_batch = []
                        for reward, p in zip(reward_batch, p_batch):
                            reward_scaled = util_agent.scale_reward(
                                reward, pre_reward=p.pre_reward
                            )
                            if reward_scaled > self.reward_threshold:
                                reward_scaled_batch += [reward_scaled]
                            else:
                                reward_scaled_batch += [0 * reward_scaled]
                        next_state_batch = np.vstack([
                            param_agent.get_state_representation(p)
                            for p in p_batch
                        ])
                        next_state_batch = np.hstack((
                            next_state_batch,
                            deepcopy(target_latent_batch),
                        ))
                        done = False
                        if step == self.num_adapt_step - 1:
                            done = True

                        # Add data to buffer - use scaled reward - no need to save traj since not updating traj network
                        for p_ind in range(self.num_trainer):
                            infer_agent.store_transition(
                                state_batch[p_ind], action_batch[p_ind],
                                reward_scaled_batch[p_ind],
                                next_state_batch[p_ind], done, None
                            )

                        ########### Update for next step ###########

                        # Add particle to all - assign a new id - reset
                        p_batch_new = []
                        for p in p_batch:
                            p_new = deepcopy(p)
                            p_new.id = self.p_id_cnt
                            self.p_id_cnt += 1
                            p_batch_new += [p_new]

                            # Update p_all
                            self.p_all += [deepcopy(p_new)]
                        p_batch = p_batch_new

                        # Record reward progress
                        reward_scaled_sum += reward_scaled_batch

                    # Count
                    for _ in range(self.num_trainer):
                        eps_schduler.step()
                    adapt_cnt += self.num_trainer
                    pbar.update(self.num_trainer)

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

                            # Save particles
                            with open(
                                os.path.join(
                                    self.inference_model_folder, 'particles'
                                ), 'wb'
                            ) as f:
                                pickle.dump({
                                    'init_particles': self.p_init,
                                    'particles': self.p_all
                                }, f, pickle.HIGHEST_PROTOCOL)

            ##################################################################
            ###################      Meta Level        ######################
            ##################################################################
            logging.info('Iteration done.\n')

            # Record
            loss = np.mean(list(loss.values()))
            if self.use_wandb:
                wandb.log({"Inference loss": loss}, step=cnt_itr + 1,
                          commit=False)

            # Adapt particles
            if cnt_itr > 0:
                for p in self.p_init:

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
                    action = infer_agent.infer(
                        state, random_flags=[False], verbose=False
                    )[0]  # do not explore in real
                    param_agent.apply_action(action, p)
                    logging.info(
                        'Particle id {} updated param to {} with action {}.'.
                        format(p.id, p.dist, action)
                    )
