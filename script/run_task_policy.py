import os
import sys
import argparse
import time
import logging
import pretty_errors
import wandb
import matplotlib


matplotlib.use('Agg')
from omegaconf import OmegaConf

# AGENT
from agent.policy import get_policy_agent
from env import get_env, get_vec_env, get_vec_env_cfg
from adaptsim.env.util.vec_env import make_vec_envs


def main(cfg):
    ################### Logging ###################
    log_file = os.path.join(cfg.out_folder, 'log.log')
    log_fh = logging.FileHandler(log_file, mode='w+')
    log_sh = logging.StreamHandler(sys.stdout)
    log_format = '%(asctime)s %(levelname)s: %(message)s'
    # Possible levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(
        format=log_format, level='INFO', handlers=[log_sh, log_fh]
    )

    ###################### cfg ######################
    os.makedirs(cfg.out_folder, exist_ok=True)

    # Learner
    if 'learner' in cfg.policy:
        cfg.policy.learner.arch.action_dim = cfg.env.action_dim
        cfg.policy.learner.eval = cfg.policy.eval

    ###################### Env ######################
    # Common args
    env_cfg = OmegaConf.create({'render': cfg.env.render, 'dt': cfg.env.dt})

    # Add
    env_cfg = OmegaConf.merge(env_cfg, cfg.env.specific)

    # Initialize workers
    logging.info("== Environment Information ==")
    venv = make_vec_envs(
        env_type=get_env(cfg.env.name),
        seed=cfg.seed,
        num_env=cfg.num_cpus,
        device=cfg.device,
        vec_env_type=get_vec_env(cfg.env.name),
        vec_env_cfg=get_vec_env_cfg(cfg.env.name, cfg.env),
        **env_cfg,
    )

    # Agent
    logging.info("== Agent Information ==")
    agent = get_policy_agent(cfg.policy.name)(cfg.policy, venv)
    if 'learner' in cfg.policy and cfg.policy.name != 'Blackbox':
        logging.info(
            'Total parameters in policy: {}'.format(
                sum(
                    p.numel()
                    for p in agent.learner.parameters
                    if p.requires_grad
                )
            )
        )
        logging.info(
            "We want to use: {}, and Agent uses: {}".format(
                cfg.device, agent.learner.device
            )
        )

    # Learn
    start_time = time.time()
    if cfg.policy.eval:
        logging.info("\n== Evaluating ==")
        agent.evaluate(
            policy_path=cfg.policy.policy_path,
            num_episode=cfg.policy.num_eval_episode, verbose=True
        )
    else:
        logging.info("\n== Learning ==")
        agent.learn(
            policy_path=cfg.policy.policy_path,
            optimizer_state=cfg.policy.optim_path, verbose=False
        )
    logging.info('\nTime used: {:.1f}'.format(time.time() - start_time))


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    main(cfg)
