from .util_adapt_dp import UtilAdaptDP
from .util_adapt_dp_linearized import UtilAdaptDPLinearized
from .util_adapt_pendulum import UtilAdaptPendulum
from .util_adapt_push import UtilAdaptPush
from .util_adapt_scoop import UtilAdaptScoop
from .util_adapt_acrobot import UtilAdaptAcrobot


util_agent_dict = {
    'DP': UtilAdaptDP,
    'DP-Linearized': UtilAdaptDPLinearized,
    'Pendulum': UtilAdaptPendulum,
    'Push': UtilAdaptPush,
    'Scoop': UtilAdaptScoop,
    'Acrobot': UtilAdaptAcrobot
}
