import os
import argparse
import numpy as np
import pickle
from omegaconf import OmegaConf

from util.numeric import sample_uniform


class Generator():

    def __init__(self, rng):
        """
        """
        self.rng = rng

        # Initialize link and joint holders - assume max 3 links for now
        self.reset()

    def reset(self):
        return

    def update_path(self, postfix='v0'):
        # Set path
        self.current_data_path = os.path.join('data', postfix)
        if not os.path.exists(self.current_data_path):
            os.makedirs(self.current_data_path)

    def update_property(self, **kwargs):
        # not good design
        self.__dict__.update(kwargs)

    def generate(self, num_task):

        save_tasks = []
        for _ in range(num_task):
            task = OmegaConf.create()

            # Fill in task
            m1 = float(sample_uniform(self.rng, self.m1_range))
            m2 = float(sample_uniform(self.rng, self.m2_range))
            b1 = float(sample_uniform(self.rng, self.b1_range))
            b2 = float(sample_uniform(self.rng, self.b2_range))

            task.true_m = [m1, m2]
            task.true_b = [b1, b2]

            # Add to task
            save_tasks += [task]

        path = os.path.join(self.current_data_path, str(num_task) + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(save_tasks, f)
        return path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", type=str)
    args = parser.parse_args()
    if args.cfg_file == 'None':
        cfg = OmegaConf.create()
        cfg.m1_range = [1.0, 1.0]
        cfg.m2_range = [1.0, 1.0]
        cfg.b1_range = [1.0, 1.0]
        cfg.b2_range = [1.0, 1.0]
        cfg.parent_name = 'linearized/dp/'
        cfg.num_task = 1000
        cfg.np_seed = 42
    else:
        cfg = OmegaConf.load(args.cfg_file)

    # Generate
    rng = np.random.default_rng(seed=cfg.np_seed)
    generator = Generator(rng=rng)
    generator.update_path(postfix=cfg.parent_name)
    generator.update_property(
        m1_range=cfg.m1_range,
        m2_range=cfg.m2_range,
        b1_range=cfg.b1_range,
        b2_range=cfg.b2_range,
    )
    generator.generate(num_task=cfg.num_task)
