from .policy_linearized import PolicyLinearized
from .policy_value import PolicyValue
from .policy_value_dyn_fit import PolicyValueDynFit


policy_agent_dict = {
    'Linearized': PolicyLinearized,
    'Value': PolicyValue,
    'ValueDynFit': PolicyValueDynFit,
}
