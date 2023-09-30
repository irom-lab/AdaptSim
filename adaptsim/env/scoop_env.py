import os
import numpy as np
import logging
import scipy.interpolate
from omegaconf import OmegaConf
from pydrake.all import RotationMatrix, RollPitchYaw, RigidTransform, FixedOffsetFrame, CollisionFilterDeclaration

from adaptsim.env.panda_env import PandaEnv
from adaptsim.panda.util import AddMultibodyTriad


GEOM_ONEHOT = {'ellipsoid': 0, 'cylinder': 1, 'box': 2}
GEOM_ONEHOT_INV = {0: 'ellipsoid', 1: 'cylinder', 2: 'box'}


class ScoopEnv(PandaEnv):

    def __init__(
        self,
        dt=0.005,
        render=False,
        visualize_contact=False,
        hand_type='panda',
        diff_ik_filter_hz=200,
        contact_solver='sap',
        panda_joint_damping=1.0,
        table_type='normal',
        flag_disable_rate_limiter=True,
        num_link=1,
        veggie_x=0.68,
        mid_frame_timestamp=[0.7],
        traj_type='absolute',  # 'absolute, relative_to_tip'
        reward_type='simple',  # 'simple', 'full'
        max_pad_center_dist=0.05,
        pad_center_reward_coeff=0.2,
        **kwargs,
    ):
        super(ScoopEnv, self).__init__(
            dt=dt,
            render=render,
            visualize_contact=visualize_contact,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
            table_type=table_type,
            flag_disable_rate_limiter=flag_disable_rate_limiter,
        )
        self.num_link = num_link
        self.veggie_template_path = os.path.join(
            'asset/veggie_template/veggie_{}link.sdf'.format(num_link)
        )
        self.veggie_x = veggie_x  # for scooping direction - assume veggies around this position in x
        self.finger_init_pos = 0.04
        self.flag_init_reset = True
        self.mid_frame_timestamp = mid_frame_timestamp
        if traj_type == 'relative_to_tip':
            self.obs_func = self._get_veggie_xy_relative_to_tip
        elif traj_type == 'absolute':
            self.obs_func = self._get_veggie_xy
        else:
            raise 'Unknown traj type {}'.format(traj_type)
        self.max_pad_center_dist = max_pad_center_dist
        self.pad_center_reward_coeff = pad_center_reward_coeff
        if reward_type not in ['simple', 'full']:
            raise 'Unknown reward type {}'.format(reward_type)
        self.reward_type = reward_type

        # Fixed dynamics parameter
        self.veggie_hc_dissipation = 1.0  # no effect?
        self.veggie_sap_dissipation = 0.1
        self.veggie_hydro_resolution = 0.01

        # Set default task
        self.task = OmegaConf.create()
        self.task.obj_num = 1
        self.task.sdf = 'asset/veggies/sample_ellipsoid.sdf'
        self.task.obj_mu = 0.4
        self.task.obj_modulus = 5
        self.task.obj_x = [0.70, 0.68, 0.68]
        self.task.obj_y = [0, -0.02, 0.02]
        self.task.obj_z = [0.01, 0.02, 0.03]

    def reset_task(self, task):
        return NotImplementedError

    @property
    def parameter(self):
        # for adaptation or policy training
        geom_type_all = [
            self.task.sdf_cfg['link' + str(link_ind)]
            for link_ind in range(self.num_link)
        ]
        geom_onehot_all = [GEOM_ONEHOT[name] for name in geom_type_all]
        # return [self.task.obj_mu, self.task.obj_modulus] + geom_onehot_all # x0, y0, z0, m0
        return [self.task.obj_mu, self.task.obj_modulus
               ] + geom_onehot_all + [self.task.sdf_cfg['z0']]  #!

    def load_objects(self,):

        # Load spatula
        spatula_path = 'asset/spatula_long/spatula_oxo_nylon_square_issue7322_low.sdf'
        self.spatula_model_index, self.spatula_body_index = \
            self.station.AddModelFromFile(spatula_path, name='spatula')
        self.spatula_base = self.plant.get_body(
            self.spatula_body_index[0]
        )  # only base
        self.spatula_base_frame = self.plant.GetFrameByName(
            "origin", self.spatula_model_index
        )
        self.spatula_blade_frame = self.plant.GetFrameByName(
            "spatula_blade_origin_frame", self.spatula_model_index
        )
        self.spatula_grasp_frame = self.plant.GetFrameByName(
            "spatula_grasp_frame", self.spatula_model_index
        )
        self.spatula_tip_frame = self.plant.GetFrameByName(
            "spatula_tip_frame", self.spatula_model_index
        )
        if self.render:
            AddMultibodyTriad(
                self.spatula_blade_frame,
                self.sg,
                length=.01,
                radius=0.0002,
            )

        # Load veggie template with fixed number of links (bodies to be replaced later) - save body and frame ids
        self.veggie_body_all = {}  # list of lists, in the order of links
        self.veggie_frame_all = {}
        for ind in range(self.task.obj_num):
            veggie_model_index, veggie_body_indice = \
                self.station.AddModelFromFile(
                    self.veggie_template_path,
                    name='veggie'+str(ind),
                )
            self.veggie_body_all[ind] = [
                self.plant.get_body(index) for index in veggie_body_indice
            ]  # all links
        self.veggie_base_all = [b[0] for b in self.veggie_body_all.values()]
        self.veggie_default_inertia = self.veggie_body_all[0][
            0].default_spatial_inertia()  # any link works

        # Add a generic frame for veggies - fixed to table
        self.T_veggie = np.array([self.veggie_x, 0, self.table_offset])
        self.veggie_fixed_frame = self.plant.AddFrame(
            FixedOffsetFrame(
                "veggie_fixed_frame", self.plant.world_frame(),
                RigidTransform(self.T_veggie)
            )
        )

    def reset(self, task=None):
        """
        Call parent to reset arm and gripper positions (build if first-time). Reset veggies and task. Do not initialize simulator.
        """
        task = super().reset(task)

        # Get new context
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        context_inspector = query_object.inspector()

        # Change global params - time step for both plant and controller_plant? seems impossible
        # point - plant.set_penetration_allowance (stiffness of normal penalty forces)
        # point/hydro - plant.set_stiction_tolerance (threshold for sliding)

        # Set veggie geometry - bodies is a list of bodies for one piece
        if not self.visualize_contact:
            sdf_cfg = task.sdf_cfg
            sg_geometry_ids = context_inspector.GetAllGeometryIds()
            for _, bodies in self.veggie_body_all.items():
                for body_ind, body in enumerate(bodies):

                    # Get frame id  #? better way? - quit if not found in sg
                    old_geom_id = self.plant.GetCollisionGeometriesForBody(
                        body
                    )[0]
                    if old_geom_id not in sg_geometry_ids:
                        raise "Error: plant geometry id not found in scene graph, likely an issue with swapping geometry ids!"
                    frame_id = context_inspector.GetFrameId(old_geom_id)

                    # Replace body - this does not change the body mass/inertia
                    if 'link' + str(body_ind) not in sdf_cfg:
                        geom_type = None  # flag for no link
                        x_dim, y_dim, z_dim, x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    else:
                        body_str = str(body_ind)
                        geom_type = sdf_cfg['link' + body_str]
                        x_dim = sdf_cfg['x' + body_str]
                        y_dim = sdf_cfg['y' + body_str]
                        z_dim = sdf_cfg['z' + body_str]
                        if body_ind == 0:
                            x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0
                        else:
                            x = sdf_cfg['x0' + body_str]
                            y = sdf_cfg['y0' + body_str]
                            z = sdf_cfg['z0' + body_str]
                            roll = sdf_cfg['roll0' + body_str]
                            pitch = sdf_cfg['pitch0' + body_str]
                            yaw = sdf_cfg['yaw0' + body_str]

                    flag_replace = self.replace_body(
                        context=sg_context,
                        context_inspector=context_inspector,
                        body=body,
                        frame_id=frame_id,
                        geom_type=geom_type,
                        x_dim=x_dim,
                        y_dim=y_dim,
                        z_dim=z_dim,
                        x=x,
                        y=y,
                        z=z,
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        visual_name='link' + body_str + '_visual',
                        collision_name='link' + body_str + '_collision',
                    )

                    # Change body dynamics - if new geometry is added
                    if flag_replace:
                        self.set_obj_dynamics(
                            context_inspector,
                            sg_context,
                            body,
                            hc_dissipation=self.veggie_hc_dissipation,
                            sap_dissipation=self.veggie_sap_dissipation,
                            mu=max(0.01, task.obj_mu),
                            hydro_modulus=max(3, task.obj_modulus),
                            hydro_resolution=self.veggie_hydro_resolution,
                            compliance_type='compliant',
                        )

                        # First, revert inertia back to origin
                        body.SetSpatialInertiaInBodyFrame(
                            plant_context, self.veggie_default_inertia
                        )

                        # Change mass
                        body.SetMass(plant_context, sdf_cfg['m' + body_str])

                        # Shift COM to body center (frame is always at the center of base link)
                        if body_ind > 0:
                            inertia = body.CalcSpatialInertiaInBodyFrame(
                                plant_context
                            )
                            inertia = inertia.Shift([-x, -y, -z])
                            body.SetSpatialInertiaInBodyFrame(
                                plant_context, inertia
                            )
                        # print(body.CalcSpatialInertiaInBodyFrame(plant_context).CopyToFullMatrix6())
                        # print(body.CalcSpatialInertiaInBodyFrame(plant_context).IsPhysicallyValid())

                # Exclude collision within body - use contac
                body_geometry_set = self.plant.CollectRegisteredGeometries(
                    bodies
                )
                self.sg.collision_filter_manager(sg_context).Apply(
                    CollisionFilterDeclaration().
                    ExcludeWithin(body_geometry_set)
                )
        self.veggie_radius = task.obj_radius

        # Set table properties
        if self.flag_init_reset:
            self.set_obj_dynamics(
                context_inspector,
                sg_context,
                self.table_body,
                hc_dissipation=self.veggie_hc_dissipation,
                sap_dissipation=self.veggie_sap_dissipation,
                mu=max(0.01, task.obj_mu),
                hydro_modulus=5,
                hydro_resolution=0.1,  # does not matter
                compliance_type='rigid'
            )

        # Move spatula away from veggies
        self.set_body_pose(
            self.spatula_base, plant_context, p=[0.3, 0, 0.01], rpy=[0, 0, 0]
        )
        self.set_body_vel(self.spatula_base, plant_context)

        # Set veggie pose from task
        for ind, veggie_base in enumerate(self.veggie_base_all):
            self.set_body_pose(
                veggie_base, plant_context, p=[
                    task.obj_x[ind],
                    task.obj_y[ind],
                    task.obj_z[ind] + self.table_offset,
                ], rpy=[0, 0, 0]
            )
            self.set_body_vel(veggie_base, plant_context)

        # Set spatula
        if self.flag_init_reset:
            self.set_obj_dynamics(
                context_inspector,
                sg_context,
                self.spatula_base,
                hc_dissipation=self.veggie_hc_dissipation,
                sap_dissipation=self.veggie_sap_dissipation,
                mu=max(0.01, task.obj_mu),
                hydro_modulus=5,
                hydro_resolution=0.1,  # does not matter
                compliance_type='rigid'
            )

        # Get fixed transforms between frames
        if self.flag_init_reset:
            self.p_spatula_tip_to_spatula_base = self.spatula_base_frame.CalcPose(
                plant_context, self.spatula_tip_frame
            )
            self.p_spatula_tip_to_spatula_grasp = self.spatula_grasp_frame.CalcPose(
                plant_context, self.spatula_tip_frame
            )
        self.flag_init_reset = False

        ######################## Observation ########################

        context = self.simulator.get_mutable_context()
        station_context = self.station.GetMyContextFromRoot(context)
        return self._get_obs(plant_context, station_context)

    def step(self, action):
        """
        Initialize simulator and then execute open-loop.
        """
        # Get new context
        context = self.simulator.get_mutable_context()
        station_context = self.station.GetMyContextFromRoot(context)
        plant_context = self.plant.GetMyContextFromRoot(context)
        controller_plant_context = self.controller_plant.GetMyContextFromRoot(
            context
        )
        # sg_context = self.sg.GetMyMutableContextFromRoot(context)
        # query_object = self.sg.get_query_output_port().Eval(sg_context)
        # inspector = query_object.inspector()

        # Extract action
        s_pitch_init_raw, pd_1_raw, s_x_veggie_tip = action
        s_yaw = 0
        xd_1 = 0.6
        pd_2 = -0.6

        # add noise
        s_pitch_init = s_pitch_init_raw + (
            self.rng.random() - 0.5
        ) * 2 * 0.012  #10%
        pd_1 = pd_1_raw + (self.rng.random() - 0.5) * 2 * 0.04  #10%

        # Veggie to tip
        R_VT = RotationMatrix(RollPitchYaw(0, 0, s_yaw))
        p_VT = [s_x_veggie_tip - self.veggie_radius, 0., 0.]
        p_T = self.T_veggie + p_VT

        # Reset spatula between gripper - in real, we can make a holder for spatula, thus ensures
        # grasp always at the same pose on the spatula. Make sure the tip of spatula touches the
        # table when the spatula is tilted
        R_T = R_VT.multiply(RotationMatrix(RollPitchYaw(0, s_pitch_init, 0)))
        R_S = R_T.multiply(self.p_spatula_tip_to_spatula_base.rotation())
        p_S = p_T + R_T.multiply(
            self.p_spatula_tip_to_spatula_base.translation()
        )
        self.set_body_pose(self.spatula_base, plant_context, p=p_S, rm=R_S)
        self.set_body_vel(self.spatula_base, plant_context)

        # Finally, EE
        R_E = R_VT.multiply(self.R_TE)
        grasp_pose = self.spatula_grasp_frame.CalcPoseInWorld(plant_context)
        if self.hand_type == 'panda_foam':
            p_E = grasp_pose.translation() + np.array([0, 0, 0.16])
        elif self.hand_type == 'panda':
            p_E = grasp_pose.translation() + np.array([0, 0, 0.10])
        else:
            raise "Unknown hand type!"
        qstar = self.ik(
            plant_context, controller_plant_context, T_e=p_E, R_e=R_E
        )
        self.set_arm(plant_context, qstar)
        self.set_gripper(plant_context, self.finger_init_pos)

        # Initialize state interpolator/integrator
        self.reset_state(plant_context, context)

        # Reset simulation
        sim_context = self.simulator.get_mutable_context()
        sim_context.SetTime(0.)
        self.simulator.Initialize()

        # Grab initial observation
        station_context = self.station.GetMyContextFromRoot(context)

        ######################## Trajectory ########################
        hand_width_command = -0.2
        obs_all = []
        veggie_xy_all = []

        # Collect info
        info = {}
        info['action'] = action
        info['parameter'] = self.parameter
        info['reward'] = 0
        info['error'] = False
        reward = 0
        done = True  # always done: single step

        # Close gripper
        num_t_init = int(0.2 / self.dt)
        for t_ind in range(1, num_t_init):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            # hand_position_measure = self.hand_state_measure_port.Eval(station_context)
            # hand_force_measure = self.hand_force_measure_port.Eval(station_context)
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Keep spatula in place
            self.set_body_pose(self.spatula_base, plant_context, p=p_S, rm=R_S)
            self.set_body_vel(self.spatula_base, plant_context)

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                info['error'] = True
                return obs_all, reward, done, info

        # First frame:
        veggie_xy_all += [self.obs_func(plant_context, randomize=True)]

        # Time for spline
        tn = [0, 0.25, 0.1, 0.25, 0.1, 0.3]
        tn = np.cumsum(tn)
        ts = np.arange(0, tn[-1], self.dt)

        # Spline for x direction
        xd = np.zeros((6))
        xd[1] = xd_1
        xd[2] = xd_1 + 0.1
        xd[3] = 0.2
        poly_xd = scipy.interpolate.CubicSpline(tn, xd, bc_type='clamped')
        xds = poly_xd(ts)

        # Spline for pitch direction
        pitchd = np.zeros((6))
        pitchd[1] = pd_1
        pitchd[2:4] = pd_2
        poly_pitchd = scipy.interpolate.CubicSpline(
            tn, pitchd, bc_type='clamped'
        )
        pitchds = poly_pitchd(ts)

        # Spline for z direction
        zd = [0, -0.01, -0.01, 0.01, 0, 0]
        poly_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
        zds = poly_zd(ts)

        # Go through trajectory
        num_t = len(ts)
        mid_frame_cnt = 0
        for t_ind in range(0, num_t):
            V_G = np.array([0, pitchds[t_ind], 0, xds[t_ind], 0,
                            zds[t_ind]]).reshape(6, 1)

            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, V_G)
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )
            # panda_desired_state = self.panda_desired_state_port.Eval(station_context)

            # contact_results_list = self.contact_results_output_port.Eval(plant_context)
            # for contact_ind in range(contact_results_list.num_hydroelastic_contacts()):
            #     he_contact_result = contact_results_list.hydroelastic_contact_info(contact_ind)
            #     surface = he_contact_result.contact_surface()
            #     geometry_A_index = surface.id_M()
            #     geometry_B_index = surface.id_N()
            #     # print(he_contact_result.contact_force(), he_contact_result.contact_point())
            #     print(inspector.GetName(geometry_A_index), inspector.GetName(geometry_B_index))

            # Simulate forward
            t = (t_ind+num_t_init) * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                info['error'] = True
                logging.info(f'Sim error at time {t}!')
                return obs_all, reward, done, info

            # Mid frame:
            if abs(
                t_ind * self.dt - self.mid_frame_timestamp[mid_frame_cnt]
            ) < self.dt * 0.1:
                veggie_xy_all += [self.obs_func(plant_context, randomize=True)]
                mid_frame_cnt = min(
                    len(self.mid_frame_timestamp) - 1, mid_frame_cnt + 1
                )

        # Final frame:
        veggie_xy_all += [self.obs_func(plant_context, randomize=True)]

        # Get more info
        info['veggie_xy_all'] = veggie_xy_all

        # Get reward - veggie piece on spatula - spatula_blade_frame is at the center of the pad
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        station_context = self.station.GetMyContextFromRoot(context)
        pose_blade = self.spatula_blade_frame.CalcPoseInWorld(plant_context)
        for veggie_body in self.veggie_base_all:
            veggie_pose = self.plant.EvalBodyPoseInWorld(
                plant_context, veggie_body
            )
            blade_veggie = pose_blade.InvertAndCompose(veggie_pose)

            # veggie higher than pad in pad frame -> on spatula
            if blade_veggie.translation()[2] > 0.001:

                if self.reward_type == 'simple':
                    reward += 1

                elif self.reward_type == 'full':
                    # Get distance from pad center to veggie
                    dist_blade_veggie_xy = np.linalg.norm(
                        blade_veggie.translation()[:2]
                    )
                    norm_pad_center_dist = dist_blade_veggie_xy / self.max_pad_center_dist
                    norm_pad_center_dist = np.clip(norm_pad_center_dist, 0, 1)

                    # reward is 1 for veggie on pad then minus weighted distance from center
                    reward += 1 - norm_pad_center_dist * self.pad_center_reward_coeff

            # no reward if drops off spatula. We can be expressive here, but not sure if this helps
        reward /= len(self.veggie_base_all)

        info['reward'] = reward
        return obs_all, reward, done, info

    def _get_veggie_xy(self, plant_context, randomize=False):
        veggie_pose_all = []
        veggie_body_ind_all = np.arange(len(self.veggie_base_all))
        if randomize:
            self.rng.shuffle(veggie_body_ind_all)
        for ind in veggie_body_ind_all:
            veggie_body = self.veggie_base_all[ind]
            veggie_pose_all += [
                self.plant.EvalBodyPoseInWorld(plant_context, veggie_body)
            ]
        return np.hstack([p.translation()[:2] for p in veggie_pose_all]
                        ).astype(np.single)

    def _get_veggie_xy_relative_to_tip(self, plant_context, randomize=False):
        veggie_pose_all = []
        veggie_body_ind_all = np.arange(len(self.veggie_base_all))
        if randomize:
            self.rng.shuffle(veggie_body_ind_all)
        for ind in veggie_body_ind_all:
            veggie_body = self.veggie_base_all[ind]
            veggie_pose = self.plant.EvalBodyPoseInWorld(
                plant_context, veggie_body
            )
            pose_tip = self.spatula_tip_frame.CalcPoseInWorld(plant_context)
            veggie_tip = veggie_pose.InvertAndCompose(pose_tip)
            veggie_pose_all += [veggie_tip]
        return np.hstack([p.translation()[:2] for p in veggie_pose_all]
                        ).astype(np.single)

    def _get_obs(self, plant_context=None, station_context=None):
        # if self.flag_use_camera:
        #     color_image = self.color_image_port.Eval(station_context
        #                                             ).data[:, :, :3]  # HxWx4
        #     color_image = np.transpose(color_image, [2, 0, 1])
        #     label_image = np.squeeze(
        #         self.label_image_port.Eval(station_context).data
        #     )
        #     label_image = np.uint8(label_image)[
        #         None]  # separate label for different bodies within one model
        #     image = np.vstack((color_image, label_image))
        #     return image
        # else:
        return self._get_veggie_xy(plant_context)
