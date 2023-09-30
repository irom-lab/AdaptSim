import torch

from adaptsim.util.numeric import normalize


class UtilPolicyPush():

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.lb = torch.tensor(list(cfg.lb))
        self.ub = torch.tensor(list(cfg.ub))
        self.belief = None

    def get_append(self, cur_tasks):
        """
        Normalize sim params.
        """
        task_rep_all = []
        for task in cur_tasks:
            if self.belief is not None:
                task_rep = torch.tensor(self.belief)
            else:
                task_rep = torch.tensor([
                    task.obj_mu, task.obj_modulus, task.obj_com_x,
                    task.obj_com_y
                ])
            task_rep = normalize(task_rep, self.lb, self.ub)
            task_rep_all += [task_rep]
        task_rep_all = torch.vstack(task_rep_all).float().to(self.device)
        return task_rep_all

    def set_belief(self, belief):
        """
        Use mean of the uniform distribution as belief.
        """
        # obj_mu = np.mean(param['OBJ_MU'])
        # obj_modulus = np.mean(param['OBJ_MODULUS'])
        # obj_com_x = np.mean(param['OBJ_COM_X'])
        # obj_com_y = np.mean(param['OBJ_COM_Y'])
        # self.belief = [obj_mu, obj_modulus, obj_com_x, obj_com_y]
        # logging.info('Setting belief {}'.format(self.belief))
        self.belief = belief

    def remove_belief(self):
        self.belief = None
