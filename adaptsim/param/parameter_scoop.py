import logging
from copy import deepcopy

from adaptsim.param.parameter_base import ParameterBase
from adaptsim.util.numeric import sample_uniform


GEOM_ONEHOT = {'ellipsoid': 0, 'cylinder': 1, 'box': 2}
GEOM_ONEHOT_INV = {0: 'ellipsoid', 1: 'cylinder', 2: 'box'}
GEOM_ONEHOT_MIN = 0
GEOM_ONEHOT_MAX = 2


class ParameterScoop(ParameterBase):

    def __init__(self, cfg):
        """
        Managing the simulation parameters in scooping example. Do not save sdf 
        file for now, which means assuming the number of links of each veggie 
        is fixed during adaptation training, so that reset can be done with 
        specified values only without new sdf.
        """
        super().__init__(cfg)
        self.num_link = cfg.num_link
        self.obj_num = cfg.obj_num
        self.obj_x_range = cfg.obj_x_range
        self.obj_y_range = cfg.obj_y_range

    # === Task generation ===
    def generate_task_wid(
        self, num_task, particle, num_goal_per_task=1, **kwargs
    ):
        num_task_param = int(num_task / num_goal_per_task)
        tasks = super().generate_task_wid(num_task_param, particle, **kwargs)
        tasks = [
            deepcopy(task) for task in tasks for _ in range(num_goal_per_task)
        ]
        for task in tasks:
            task = self.process_link_joint_height(task)
        return tasks

    def generate_task_ood(self, num_task, particle):
        #! assume 1 target in omega?
        tasks = super().generate_task_ood(num_task, particle)
        for task in tasks:
            task = self.process_link_joint_height(task)
        return tasks

    def process_link_joint_height(self, task):
        """Also samples initial pos of the pieces"""

        # oobj_mu, obj_modulus, obj_x, obj_y, obj_z already done

        def get_volumn(x, y, z, link_type):
            if link_type == 'ellipsoid':
                return 4.1888 * x * y * z
            elif link_type == 'cylinder':
                return 3.1416 * (x**2) * (2*z)
            elif link_type == 'box':
                return (2*x) * (2*y) * (2*z)
            else:
                logging.error('Unknown link type when getting volume!')
                raise

        sdf_cfg = {}
        for link_id in range(self.num_link):

            key_name = f'link{link_id}'
            geom_id = max(
                GEOM_ONEHOT_MIN, min(GEOM_ONEHOT_MAX, round(task[key_name]))
            )
            link_type = GEOM_ONEHOT_INV[geom_id
                                       ]  # 0: ellipsoid, 1: cylinder, 2: box
            sdf_cfg[key_name] = link_type
            x = task[f'x{link_id}']
            y = task[f'y{link_id}']
            z = task[f'z{link_id}']
            sdf_cfg[f'x{link_id}'] = x
            sdf_cfg[f'y{link_id}'] = y
            sdf_cfg[f'z{link_id}'] = z

            # Sample mass
            link_mass = task.obj_density * 1e3 * get_volumn(
                x, y, z, link_type
            )  # convert from g/cm^3 to kg/m^3
            sdf_cfg[f'm{link_id}'] = link_mass

        for joint_id in range(1, self.num_link):
            # joint_id==1 for joint between link0 and link1

            # Check if joint exists
            key_name = 'joint0{}'.format(joint_id)
            if key_name not in sdf_cfg.keys():
                continue

            # Use ratio to dimensions of the first link
            link_x = sdf_cfg['x0']
            link_y = sdf_cfg['y0']
            link_z = sdf_cfg['z0']
            x = task[key_name + '_x'] * link_x
            y = task[key_name + '_y'] * link_y
            z = task[key_name + '_z'] * link_z
            sdf_cfg[f'x0{joint_id}'] = x
            sdf_cfg[f'y0{joint_id}'] = y
            sdf_cfg[f'z0{joint_id}'] = z
            sdf_cfg[f'roll0{joint_id}'] = task[key_name + '_roll']
            sdf_cfg[f'pitch0{joint_id}'] = task[key_name + '_pitch']
            sdf_cfg[f'yaw{joint_id}'] = task[key_name + '_yaw']

        # Use fixed number of pieces
        task.obj_num = self.obj_num

        # Sample initial pos
        task.obj_x = sample_uniform(
            self.rng, self.obj_x_range, size=task.obj_num
        )
        task.obj_y = sample_uniform(
            self.rng, self.obj_y_range, size=task.obj_num
        )
        task.obj_z = [
            sdf_cfg['z0'] * ind + sdf_cfg['z0'] + 0.0001
            for ind in range(task.obj_num)
        ]

        # for initial
        task.obj_radius = sdf_cfg['x0']

        # Fill in sdf cfg
        task.sdf_cfg = sdf_cfg
        return task
