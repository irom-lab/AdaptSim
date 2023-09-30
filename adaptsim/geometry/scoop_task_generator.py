import os
import argparse
import numpy as np
from xmacro.xmacro import XMLMacro
from pathlib import Path
import pickle
from omegaconf import OmegaConf

from adaptsim.geometry.joint import Joint
from adaptsim.geometry.ellipsoid import Ellipsoid
from adaptsim.geometry.cylinder import Cylinder
from adaptsim.geometry.box import Box
from adaptsim.util.numeric import sample_integers, sample_uniform


class ScoopTaskGenerator():

    def __init__(
        self,
        rng,
        sdf_template_path='data/veggie_template/veggie_base.sdf.xacro',
    ):
        """Right now only supports fixed cfg of shape primitive (link0: ellipsoid, link1: box, link2: none) for all samples. We can call this class with different cfg to generate mixture distributiotns, with each mixture a fixed cfg of shape primitives. 
           Or we sample discrete cfg in this class too.
        """
        self.rng = rng

        # Template
        self.sdf_template_path = sdf_template_path

        # Initialize link and joint holders - assume max 3 links for now
        self.reset()

    def reset(self):
        self.link_all = [[], [], []]  # TODO: use max_possible_link
        self.link_name_all = [[], [], []]
        self.joint_all = [None, None]

    def update_path(self, postfix='veggie_v0'):
        # Set path
        self.current_data_path = os.path.join('data', postfix)
        self.current_sdf_path = os.path.join('asset/veggies', postfix)
        if not os.path.exists(self.current_data_path):
            os.makedirs(self.current_data_path)
        if not os.path.exists(self.current_sdf_path):
            os.makedirs(self.current_sdf_path)

    def update_property(self, **kwargs):
        self.__dict__.update(kwargs)

    def update_link(self, ind, type, **kwargs):
        if type == 'ellipsoid':
            type_class = Ellipsoid
        elif type == 'cylinder':
            type_class = Cylinder
        elif type == 'box':
            type_class = Box
        else:
            raise NameError

        self.link_all[ind] += [type_class(**kwargs)]

        # Save name
        self.link_name_all[ind] += [type]

    def update_joint(self, ind, **kwargs):
        self.joint_all[ind] = Joint(**kwargs)  # TODO

    def generate(self, num_task, num_link, use_omegaconf=True):

        xmacro = XMLMacro()
        task_id = 0
        save_tasks = []
        while task_id < num_task:
            if use_omegaconf:
                task = OmegaConf.create()
                sdf_cfg = OmegaConf.create()
            else:
                task = {}
                sdf_cfg = {}
            sdf_path = os.path.join(
                self.current_sdf_path,
                str(task_id) + '.sdf'
            )

            # Number - #TODO: noise in sim parameters among objects of the same task?
            obj_num = int(sample_integers(self.rng, self.obj_num_range))

            # Loop thru links
            for ind, (link_choice, link_choice_name) in enumerate(
                zip(self.link_all, self.link_name_all)
            ):

                if len(link_choice) > 0:
                    link_ind = self.rng.choice(range(len(link_choice)))
                    link = link_choice[link_ind]
                    link_name = link_choice_name[link_ind]
                else:
                    break

                ind = str(ind)
                link_cfg = link.sample()
                sdf_cfg["link" + ind] = link_name
                sdf_cfg["m" + ind] = link_cfg.m
                sdf_cfg["x" + ind] = link_cfg.x
                sdf_cfg["y" + ind] = link_cfg.y
                sdf_cfg["z" + ind] = link_cfg.z

            # Loop thru joints - x/y/z relative to base dims
            for ind, joint in enumerate(self.joint_all):
                if joint is None:
                    break
                ind = str(0) + str(ind + 1)  # 01 or 02
                joint_cfg = joint.sample()
                sdf_cfg["x" + ind] = joint_cfg.x * sdf_cfg['x0']
                sdf_cfg["y" + ind] = joint_cfg.y * sdf_cfg['y0']
                sdf_cfg["z" + ind] = joint_cfg.z * sdf_cfg['z0']
                sdf_cfg["roll" + ind] = joint_cfg.roll
                sdf_cfg["pitch" + ind] = joint_cfg.pitch
                sdf_cfg["yaw" + ind] = joint_cfg.yaw

            # Generate sdf
            xmacro.set_xml_file(self.sdf_template_path)
            xmacro.generate(sdf_cfg)
            xmacro.to_file(sdf_path)

            # Fill in task
            task['sdf'] = sdf_path
            task['obj_num'] = obj_num
            task['sdf_cfg'] = sdf_cfg
            task['obj_density'
                ] = link_cfg.density  #! for now, assume same for all links
            task['obj_mu'] = sample_uniform(self.rng, self.obj_mu_range)
            task['obj_modulus'] = sample_uniform(
                self.rng, self.obj_modulus_range
            )
            task['obj_x'] = sample_uniform(
                self.rng, self.obj_x_range, size=obj_num
            )
            task['obj_y'] = sample_uniform(
                self.rng, self.obj_y_range, size=obj_num
            )
            task['obj_z'] = [(sdf_cfg['z0']) * ind + sdf_cfg['z0'] + 0.0001
                             for ind in range(obj_num)]
            # task.obj_z = [(sdf_cfg.z0*2)*ind+sdf_cfg.z0+0.0001 for ind in range(obj_num)]

            if num_link == 1:
                task['obj_radius'] = sdf_cfg['x0']
            elif num_link == 2:
                task['obj_radius'] = sdf_cfg['x01'] + sdf_cfg['x1']
            else:
                raise 'Not supported!'

            # Add to task
            save_tasks += [task]
            task_id += 1

        path = os.path.join(self.current_data_path, str(num_task) + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(save_tasks, f)
        return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)

    # Generate
    rng = np.random.default_rng(seed=cfg.np_seed)
    generator = ScoopTaskGenerator(rng=rng)
    generator.update_path(postfix=cfg.data_path)
    generator.update_property(
        obj_num_range=cfg.obj_num_range, obj_x_range=cfg.obj_x_range,
        obj_y_range=cfg.obj_y_range, obj_z_range=cfg.obj_z_range,
        obj_mu_range=cfg.obj_mu_range, obj_modulus_range=cfg.obj_modulus_range,
        obj_density_range=cfg.obj_density_range
    )

    # Link and joints
    max_possible_link = 3
    num_link = 0
    for link_ind in range(max_possible_link):
        link_type_entry = f'link_{link_ind}_type'
        if hasattr(cfg, link_type_entry):
            num_link += 1
            for link_type in cfg[link_type_entry]:
                if link_type == 'cylinder':
                    generator.update_link(
                        ind=link_ind, type=link_type, rng=rng,
                        DENSITY=cfg.obj_density_range,
                        R_DIM=cfg[f'link{link_ind}_x_radii_range'],
                        Z_DIM=cfg[f'link{link_ind}_z_radii_range']
                    )
                elif link_type == 'ellipsoid':
                    generator.update_link(
                        ind=link_ind, type=link_type, rng=rng,
                        DENSITY=cfg.obj_density_range,
                        X_DIM=cfg[f'link{link_ind}_x_radii_range'],
                        Y_DIM=cfg[f'link{link_ind}_y_radii_range'],
                        Z_DIM=cfg[f'link{link_ind}_z_radii_range']
                    )
                elif link_type == 'box':
                    generator.update_link(
                        ind=link_ind, type=link_type, rng=rng,
                        DENSITY=cfg.obj_density_range,
                        X_DIM=cfg[f'link{link_ind}_x_radii_range'],
                        Y_DIM=cfg[f'link{link_ind}_y_radii_range'],
                        Z_DIM=cfg[f'link{link_ind}_z_radii_range']
                    )
                else:
                    raise 'Unknown link type!'
                print(f'Added link {link_ind} with type {link_type}!')
            if link_ind > 0:
                generator.update_joint(
                    ind=link_ind - 1, rng=rng, X=cfg.joint_x_ratio_range,
                    Y=cfg.joint_y_ratio_range, Z=cfg.joint_z_ratio_range,
                    ROLL=[0, 0], PITCH=[0, 0], YAW=[0, 0]
                )

    generator.generate(
        num_task=cfg.num_task, num_link=num_link,
        use_omegaconf=cfg.use_omegaconf
    )
