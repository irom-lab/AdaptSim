import os
import argparse
import numpy as np
from omegaconf import OmegaConf

from adaptsim.util.numeric import sample_uniform


class PushTaskGenerator():

    def __init__(self, rng):
        """
        mu/modulus/dissipation/mass/COM.
        Do not generate new sdf.
        """
        self.rng = rng

        # Initialize link and joint holders - assume max 3 links for now
        self.reset()

    def reset(self):
        return

    def update_path(self, postfix='bottle_v0'):
        # Set path
        self.current_data_path = os.path.join('data', postfix)
        if not os.path.exists(self.current_data_path):
            os.makedirs(self.current_data_path)

    def update_property(self, **kwargs):
        # not good design
        self.__dict__.update(kwargs)

    def generate(self, num_task, use_omegaconf=True):

        def get_x_y(radius, yaw):
            return [float(np.cos(yaw) * radius), float(np.sin(yaw) * radius)]

        save_tasks = []
        for _ in range(num_task):
            if use_omegaconf:
                task = OmegaConf.create()
            else:
                task = {}

            # Fill in task
            task['obj_mu'] = float(sample_uniform(self.rng, self.obj_mu_range))
            task['obj_modulus'] = float(
                sample_uniform(self.rng, self.obj_modulus_range)
            )
            # task.obj_mass = float(sample_uniform(self.rng, self.obj_mass_range))
            task['obj_com_x'] = float(
                sample_uniform(self.rng, self.obj_com_x_range)
            )
            task['obj_com_y'] = float(
                sample_uniform(self.rng, self.obj_com_y_range)
            )

            if self.goal is not None:
                radius = float(
                    sample_uniform(self.rng, self.goal.radius_range)
                )
                yaw = float(sample_uniform(self.rng, self.goal.yaw_range))
                task['goal'] = get_x_y(radius, yaw)

            # Add to task
            save_tasks += [task]

        path = os.path.join(self.current_data_path, str(num_task) + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(save_tasks, f)
        return path


if __name__ == '__main__':

    # Notes on choosing a value:
    # Starting with the value for Young’s modulus is not unreasonable. Empirical evidence suggests that the hydroelastic modulus tends to be smaller depending on the size of the objects.
    # For large modulus values, the resolution of the representations matter more. A very high modulus will keep the contact near the surface of the geometry, exposing tessellation artifacts. A smaller modulus has a smoothing effect.
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", type=str)
    args = parser.parse_args()
    if args.cfg_file == 'None':
        cfg = OmegaConf.create()
        cfg.obj_mu_range = [0.1, 0.2]
        cfg.obj_modulus_range = [5, 7]
        cfg.obj_com_x_range = [-1e-2, 1e-2]
        cfg.obj_com_y_range = [-1e-2, 1e-2]
        cfg.parent_name = 'bottle_task/v0/'
        cfg.num_task = 2000
        cfg.np_seed = 42
    else:
        cfg = OmegaConf.load(args.cfg_file)

    # Generate
    rng = np.random.default_rng(seed=cfg.np_seed)
    generator = PushTaskGenerator(rng=rng)
    generator.update_path(postfix=cfg.parent_name)
    generator.update_property(
        #   obj_mass_range=cfg.obj_mass_range,
        obj_mu_range=cfg.obj_mu_range,
        obj_modulus_range=cfg.obj_modulus_range,
        obj_com_x_range=cfg.obj_com_x_range,
        obj_com_y_range=cfg.obj_com_y_range,
        goal=cfg.goal,
    )
    generator.generate(num_task=cfg.num_task, use_omegaconf=cfg.use_omegaconf)

    # # Dump cfg to yaml
    # cfg_path = os.path.join('data', parent_name, 'cfg.yaml')
    # cfg = OmegaConf.create()
    # cfg.num_task = num_task
    # cfg.obj_mu_range = obj_mu_range
    # cfg.obj_modulus_range = obj_modulus_range
    # cfg.obj_com_x_range = obj_com_x_range
    # cfg.obj_com_y_range = obj_com_y_range
    # with open(cfg_path, 'w') as fp:
    #     OmegaConf.save(config=cfg, f=fp)
