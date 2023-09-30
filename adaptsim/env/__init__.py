from .double_pendulum_env import DoublePendulumEnv
from .double_pendulum_linearized_env import DoublePendulumLinearizedEnv
from .push_env import PushEnv
from .push_overlap_env import PushOverlapEnv
from .scoop_env import ScoopEnv
from .pendulum_env import PendulumEnv
from .acrobot_env import AcrobotEnv
from .vec_env import VecEnvDP, VecEnvDPLinearized, VecEnvPush, VecEnvScoop, VecEnvPendulum, VecEnvAcrobot


env_dict = {
    'DP': DoublePendulumEnv,
    'DP-Linearized': DoublePendulumLinearizedEnv,
    'Acrobot': AcrobotEnv,
    'Pendulum': PendulumEnv,
    'Push': PushEnv,
    'PushOverlap': PushOverlapEnv,
    'Scoop': ScoopEnv,
}

vec_env_dict = {
    'DP': VecEnvDP,
    'DP-Linearized': VecEnvDPLinearized,
    'Acrobot': VecEnvAcrobot,
    'Pendulum': VecEnvPendulum,
    'Push': VecEnvPush,
    'PushOverlap': VecEnvPush,
    'Scoop': VecEnvScoop,
}


def get_vec_env_cfg(name, cfg_env):
    vec_env_cfg = cfg_env.specific
    return vec_env_cfg
