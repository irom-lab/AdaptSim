from .parameter_acrobot import ParameterAcrobot
from .parameter_dp import ParameterDP
from .parameter_dp_noise import ParameterDPNoise
from .parameter_push import ParameterPush
from .parameter_scoop import ParameterScoop
from .param_inference import ParamInference
from .bayessim import BayesSim


param_agent_dict = {
    'DP-Linearized': ParameterDP,
    'DP-Linearized-Noise': ParameterDPNoise,
    'Pendulum': ParameterDP,
    'Acrobot': ParameterAcrobot,
    'Push': ParameterPush,
    'Scoop': ParameterScoop,
}

infer_agent_dict = {
    'Inference': ParamInference,
    'BayesSim': BayesSim,
}