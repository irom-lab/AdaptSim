"""
Run adaptation training.

"""
import os
import sys
import argparse
import logging
import pretty_errors
import wandb
import matplotlib
from omegaconf import OmegaConf

from adaptsim.agent import agent_dict
from adaptsim.env import env_dict, vec_env_dict, get_vec_env_cfg
from adaptsim.env.util.vec_env import make_vec_envs


matplotlib.use('Agg')


def main(cfg):

    ###################### cfg ######################
    os.makedirs(cfg.out_folder, exist_ok=True)
    if cfg.wandb.entity is not None:
        wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project,
            name=cfg.wandb.name
        )
        wandb.config.update(cfg)

    ################### Logging ###################
    log_file = os.path.join(cfg.out_folder, 'log.log')
    log_fh = logging.FileHandler(log_file, mode='w+')
    log_sh = logging.StreamHandler(sys.stdout)
    log_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(
        format=log_format, level='INFO', handlers=[log_sh, log_fh]
    )

    ###################### Env ######################
    # Common args
    env_cfg = OmegaConf.create({'render': cfg.env.render, 'dt': cfg.env.dt})

    # Add
    env_cfg = OmegaConf.merge(env_cfg, cfg.env.specific)

    # Initialize multiple venv
    logging.info("== Environment Information ==")
    venv_all = []
    for trainer_id in range(cfg.num_trainer):

        # Initialize envs
        logging.info("Initializing venv for trainer {}".format(trainer_id))
        venv = make_vec_envs(
            env_type=env_dict[cfg.env.name],
            seed=cfg.seed,
            num_env=cfg.num_env_per_trainer,
            device=cfg.device,
            vec_env_type=vec_env_dict[cfg.env.name],
            vec_env_cfg=get_vec_env_cfg(cfg.env.name, cfg.env),
            **env_cfg,
        )
        venv_all += [venv]

    ###################### Agent ######################
    logging.info("== Agent Information ==")
    agent = agent_dict[cfg.agent](cfg, venv_all)

    # Run
    logging.info("== Adapting ==")
    agent.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_file)
    main(cfg)
