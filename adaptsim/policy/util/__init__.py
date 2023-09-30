from .util_policy_push import UtilPolicyPush
from .util_policy_scoop import UtilPolicyScoop
from .util_policy_dummy import UtilPolicyDummy


utility_policy_dict = {
    'Dummy': UtilPolicyDummy,
    'Push': UtilPolicyPush,
    'Scoop': UtilPolicyScoop,
}
