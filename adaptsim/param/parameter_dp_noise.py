import numpy as np

from adaptsim.param.parameter_base import ParameterBase


class ParameterDPNoise(ParameterBase):

    def __init__(self, cfg):
        """
        Managing the simulation parameters in Double Pendulum example with noise.
        """
        super().__init__(cfg)

    # === Task generation ===
    def generate_task_wid(self, num_task, particle, **kwargs):
        """
        Generate tasks by sampling from the parameter distribution.
        """
        tasks = super().generate_task_wid(num_task, particle, **kwargs)

        # Get mean - clip since BayesSim does not clip
        param_mean = np.clip(particle.mean, self.lower_bound, self.upper_bound)
        param_mean = [float(v) for v in param_mean]

        # Add mean to task - as initial policy
        for task in tasks:
            task.param_mean = param_mean

        return tasks

    def generate_task_ood(self, num_task, particle, **kwargs):
        """
        Generate tasks by sampling from the parameter distribution.
        """
        tasks = super().generate_task_ood(num_task, particle, **kwargs)

        # Get mean
        task = tasks[0]
        param_mean = [
            task.m1, task.m2, task.b2, task.b2, task.n0, task.n1, task.n2,
            task.n3, task.n4, task.n5
        ]

        # Add mean to task - for optimal reward
        for task in tasks:
            task.param_mean = param_mean
        return tasks
