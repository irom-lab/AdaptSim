# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Various ways to summarize/compress trajectories. From BayesSim"""
import torch


def summary_start(states, actions, cfg):
    """Outputs a short initial snippet of the trajectory."""
    # states, actions = pad_states_actions(states, actions, cfg.max_t)
    # num_task_batch x num_goal_per_task x num_traj_per_task x max_traj_len x obs_dim
    states = states[:, :, :, :cfg.max_t, :]
    actions = actions[:, :, :, :cfg.max_t, :]
    feats = torch.cat([states, actions], dim=-1)
    bsz = feats.shape[0]
    return feats.view(bsz, -1)  # flatten to (N,-1)


def summary_waypts_open_loop(states, actions, cfg):
    """Outputs states at fixed intervals to retain n_waypoints. Action is typically open-loop so do not pad."""
    # num_task_batch x num_goal_per_task x num_traj_per_task x max_traj_len x obs_dim
    ntraj, _, _, traj_len, _ = states.shape
    waypts = torch.linspace(
        cfg.min_t, traj_len - 1, steps=cfg.num_t, dtype=torch.long
    )  # potential   bug: some trajectory is incomplete
    states = torch.flatten(states[:, :, :, waypts, :], start_dim=1)
    actions = torch.flatten(actions[:, :, :, 0, :], start_dim=1)
    feats = torch.cat([states, actions], dim=-1)
    return feats


def summary_waypts(states, actions, cfg):
    """Outputs states and actions at fixed intervals to retain n_waypoints."""
    # num_task_batch x num_goal_per_task x num_traj_per_task x max_traj_len x obs_dim
    ntraj, _, _, traj_len, state_dim = states.shape
    waypts = torch.linspace(
        cfg.min_t, traj_len - 1, steps=cfg.num_t, dtype=torch.long
    )  # potential   bug: some trajectory is incomplete
    states = torch.flatten(states[:, :, :, waypts, :], start_dim=1)
    actions = torch.flatten(actions[:, :, :, waypts, :], start_dim=1)
    feats = torch.cat([states, actions], dim=-1)
    return feats
