"""
Managing the simulation parameters in Double Pendulum example.

"""
import numpy as np

from adaptsim.param.parameter_base import ParameterBase


class ParameterAcrobot(ParameterBase):

    def __init__(self, cfg):

        super().__init__(cfg)

    # === Task generation ===
    def generate_task_wid(self, num_task, particle, **kwargs):
        """
        Generate tasks by sampling from the parameter distribution.
        """
        tasks = super().generate_task_wid(num_task, particle, **kwargs)

        # Add param mean to task - as policy - not needed for ood task
        mean = particle.dist.mean
        if particle.fixed_dist is not None:
            mean = np.concatenate((mean, particle.fixed_dist.mean))
        for task in tasks:
            task.param_mean = [float(v) for v in mean]
        return tasks
