from .adapt import Adapt
from .adapt_trainer import AdaptTrainer
from .adapt_sysid import AdaptSysID
from .adapt_dyn_fit import AdaptDynFit
from .adapt_bayesopt import AdaptBayesOpt


agent_dict = {
    'Adapt': Adapt,
    'AdaptTrainer': AdaptTrainer,
    'AdaptSysID': AdaptSysID,
    'AdaptDynFit': AdaptDynFit,
    'AdaptBayesOpt': AdaptBayesOpt
}
