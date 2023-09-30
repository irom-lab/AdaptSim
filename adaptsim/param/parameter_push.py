import numpy as np
from scipy.stats import qmc
from copy import deepcopy

from adaptsim.param.parameter_base import ParameterBase
from adaptsim.util.numeric import sample_uniform


class ParameterPush(ParameterBase):

    def __init__(self, cfg):
        """
        Managing the simulation parameters in pushing example.
        """
        super().__init__(cfg)
        if hasattr(cfg, 'goal'):
            self.goal_radius_range = cfg.goal.radius_range
            self.goal_yaw_range = cfg.goal.yaw_range

    # === Task generation ===
    def generate_task_wid(
        self, num_task, particle, uniform_goal=False, num_goal_per_task=1,
        **kwargs
    ):
        num_task_param = int(num_task / num_goal_per_task)
        tasks = super().generate_task_wid(num_task_param, particle, **kwargs)
        tasks = [
            deepcopy(task) for task in tasks for _ in range(num_goal_per_task)
        ]
        self.add_goal(tasks, uniform_goal)
        return tasks

    def generate_task_ood(self, num_task, particle, **kwargs):
        #! assume 1 target in omega right now. num_task just differs in goal
        tasks = super().generate_task_ood(1, particle, **kwargs)
        tasks = [tasks[0] for _ in range(num_task)]
        self.add_goal(
            tasks, uniform_goal=False
        )  #? using uniform goal can be better?
        return tasks

    def add_goal(self, tasks, uniform_goal=False, num_goal_per_task=1):
        if uniform_goal:
            # radius = self.rng.choice(self.goal_radius_range)
            # yaw = self.rng.choice(self.goal_yaw_range)
            sampler = qmc.LatinHypercube(d=2)
            sample_all = sampler.random(n=len(tasks))
            l_bounds = [self.goal_radius_range[0], self.goal_yaw_range[0]]
            u_bounds = [self.goal_radius_range[1], self.goal_yaw_range[1]]
            sample_all = qmc.scale(sample_all, l_bounds, u_bounds)
            for sample, task in zip(sample_all, tasks):
                task.goal = self.get_x_y(sample[0], sample[1])
        else:
            for task in tasks:
                radius = sample_uniform(self.rng, self.goal_radius_range)
                yaw = sample_uniform(self.rng, self.goal_yaw_range)
                task.goal = self.get_x_y(radius, yaw)

    @staticmethod
    def get_x_y(radius, yaw):
        return [float(np.cos(yaw) * radius), float(np.sin(yaw) * radius)]
