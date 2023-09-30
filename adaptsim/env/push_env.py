import time
import numpy as np
from omegaconf import OmegaConf
import logging
from pydrake.all import RotationMatrix, RollPitchYaw, PiecewisePolynomial

from adaptsim.env.panda_env import PandaEnv


class PushEnv(PandaEnv):
    """
    Dynamic pushing environment in Drake

    """

    def __init__(
        self,
        dt=0.01,
        render=False,
        visualize_contact=False,
        hand_type='panda',
        diff_ik_filter_hz=500,
        contact_solver='sap',
        panda_joint_damping=200,
        table_type='normal',
        flag_disable_rate_limiter=True,
        use_goal=False,
        max_dist_bottle_goal=0.5,
        **kwargs,  # for init_range rn
    ):
        super(PushEnv, self).__init__(
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

        # Default goal
        self.use_goal = use_goal
        self._goal = np.array([0.80, 0.0])
        self._goal_base = np.array([0.45, 0])  # bottle init
        self.max_dist_bottle_goal = max_dist_bottle_goal

        # Poses
        self.finger_init_pos = 0.055
        self.bottle_initial_pos = np.array([
            0.56, 0.0, 0.05 + self.table_offset
        ])  # fixed bottle initial pos

        # Roll/pitch threshold for bottle falling - absolute value
        self.bottle_fall_roll_threshold = np.pi / 6
        self.bottle_fall_pitch_threshold = np.pi / 6

        # Set default task
        self.task = OmegaConf.create()
        self.task.obj_mass = 0.2
        self.task.obj_mu = 0.1
        self.task.obj_modulus = 7
        self.task.obj_com_x = 0.0
        self.task.obj_com_y = 0.0
        self.obj_com_z = -0.1

    @property
    def goal(self,):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value + self._goal_base

    def reset_task(self, task):
        return NotImplementedError

    @property
    def parameter(self):
        return [
            self.task.obj_mu,
            self.task.obj_modulus,
        ]

    def load_objects(self,):
        bottle_model_index, bottle_body_indice = \
            self.station.AddModelFromFile(
                "asset/bottle/bottle.sdf",
                name='bottle',
            )
        self.bottle_body = self.plant.get_body(bottle_body_indice[0])
        self.bottle_default_inertia = self.bottle_body.default_spatial_inertia(
        )

    def reset(self, task=None):
        """
        Call parent to reset arm and gripper positions (build if first-time). Reset veggies and task. Do not initialize simulator.
        """
        task = super().reset(task)

        # Reset goal if specified
        if hasattr(task, 'goal'):
            self.goal = np.array(task['goal'])

        # Get context
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        context_inspector = query_object.inspector()

        # Set table properties
        self.set_obj_dynamics(
            context_inspector,
            sg_context,
            self.table_body,
            hc_dissipation=1.0,
            sap_dissipation=0.1,
            mu=max(0.01, task.obj_mu),
            hydro_modulus=7,
            hydro_resolution=0.1,  # does not matter
            compliance_type='rigid'
        )

        # Set hand properties
        self.set_obj_dynamics(
            context_inspector,
            sg_context,
            self.hand_body,
            hc_dissipation=1.0,
            sap_dissipation=0.1,
            mu=0.3,
            hydro_modulus=6,
            hydro_resolution=0.1,  # does not matter
            compliance_type='rigid'
        )

        # Set bottle properties
        self.set_obj_dynamics(
            context_inspector,
            sg_context,
            self.bottle_body,
            hc_dissipation=1.0,
            sap_dissipation=0.1,
            mu=max(0.01, task.obj_mu),
            hydro_modulus=max(3, task.obj_modulus),
            hydro_resolution=0.01,  # matters
            compliance_type='compliant'
        )

        # First, revert inertia back to origin
        self.bottle_body.SetSpatialInertiaInBodyFrame(
            plant_context, self.bottle_default_inertia
        )

        # Next, shift the current inertia to new COM - need to shift in the opposite direction.
        inertia = self.bottle_body.CalcSpatialInertiaInBodyFrame(plant_context)
        inertia = inertia.Shift([
            -task.obj_com_x, -task.obj_com_y, -self.obj_com_z
        ])
        self.bottle_body.SetSpatialInertiaInBodyFrame(plant_context, inertia)
        # print(self.bottle_body.CalcCenterOfMassInBodyFrame(plant_context))
        # print(self.bottle_body.CalcSpatialInertiaInBodyFrame(plant_context).CopyToFullMatrix6())
        # print(self.bottle_body.CalcSpatialInertiaInBodyFrame(plant_context).IsPhysicallyValid())

        # Set mass properties - make sure do this after the inertia is reset and then set the new COM
        # self.bottle_body.SetMass(plant_context, 0.5)    # modifies SetSpatialInertiaInBodyFrame
        # print(self.bottle_body.get_mass(plant_context))

        # Reset bottle
        self.set_body_pose(
            self.bottle_body,
            plant_context,
            p=self.bottle_initial_pos,
            rpy=[0, 0, 0],
        )
        self.set_body_vel(self.bottle_body, plant_context)

        ######################## Observation ########################
        station_context = self.station.GetMyContextFromRoot(context)
        return self._get_obs(station_context)

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

        # Extract action
        vel_x, yaw = action

        # Reset EE
        if self.hand_type == 'wsg':
            R_E = RotationMatrix(RollPitchYaw(0, np.pi / 2, 0))
            R_E_fixed = RotationMatrix(RollPitchYaw(0, 0, np.pi / 2))
            T_E = [0.35, 0.20, 0.06 + self.table_offset]
            self.bottle_initial_pos = np.array([
                0.50, 0.20, 0.05 + self.table_offset
            ])
        elif self.hand_type == 'plate':
            offset = 0.15  # was 0.07
            R_E = RotationMatrix(RollPitchYaw(0, 0, 0))
            R_E_fixed = RotationMatrix(
                RollPitchYaw(0, -np.pi, np.pi / 2 + yaw)
            )
            T_E = [
                self.bottle_initial_pos[0] - offset * np.cos(yaw),
                -offset * np.sin(yaw), 0.135 + self.table_offset
            ]
        else:
            raise "Unsupported hand type!"
        R_E = R_E.multiply(R_E_fixed)
        qstar = self.ik(
            plant_context, controller_plant_context, T_e=T_E, R_e=R_E
        )
        self.set_arm(plant_context, qstar)
        if self.flag_actuate_hand:
            self.set_gripper(plant_context, self.finger_init_pos)

        # Initialize state interpolator/integrator
        self.reset_state(plant_context, context)

        # Reset simulation
        sim_context = self.simulator.get_mutable_context()
        sim_context.SetTime(0.)
        self.simulator.Initialize()
        num_t = 1

        ######################## Trajectory ########################
        bottle_T_all = []
        hand_width_command = 0.12
        vel_final = np.zeros((6, 1))
        vel_pitch = -0.8
        vel_z = 0.1
        t1 = 0.01
        t2 = 0.2
        t3 = 0.1
        t4 = 0.5  # ideally longer time

        # Collect info
        info = {}
        info['action'] = action
        info['parameter'] = self.parameter
        info['goal'] = self._goal
        info['reward'] = 0

        # Close gripper - keep arm fixed
        num_t_new = num_t + int(t1 / self.dt)
        for t_ind in range(1, num_t_new):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, np.zeros((6)))
            if self.flag_actuate_hand:
                self.hand_position_command_port.FixValue(
                    station_context, hand_width_command
                )

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info('Sim error!')
                info['error'] = True
                info['bottle_T_all'] = [self.bottle_initial_pos]
                info['bottle_T_final'] = self.bottle_initial_pos
                return np.array([]), 0, True, info
        num_t = num_t_new

        # Push forward - cartesian velocity and diff IK
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        vel_init = self.get_ee_vel(plant_context)
        vel_fast = np.array([
            -vel_pitch * np.sin(yaw), vel_pitch * np.cos(yaw), 0,
            vel_x * np.cos(yaw), vel_x * np.sin(yaw), vel_z
        ]).reshape((6, 1))
        traj_V_G = PiecewisePolynomial.FirstOrderHold([0, t2],
                                                      np.hstack(
                                                          (vel_init, vel_fast)
                                                      ))
        t_init = t
        num_t_new = num_t + int(t2 / self.dt)
        for t_ind in range(num_t, num_t_new):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(
                station_context, traj_V_G.value(t - t_init)
            )
            if self.flag_actuate_hand:
                self.hand_position_command_port.FixValue(
                    station_context, hand_width_command
                )

            # Record
            bottle_T_all += [
                self._get_bottle_pose(plant_context).translation()
            ]

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info('Sim error!')
                info['error'] = True
                info['bottle_T_all'] = bottle_T_all
                info['bottle_T_final'] = bottle_T_all[-1]
                return np.array([]), 0, True, info
        num_t = num_t_new

        # Keep speed
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        vel_init = self.get_ee_vel(plant_context)
        vel_fast = np.array([
            -vel_pitch * np.sin(yaw), vel_pitch * np.cos(yaw), 0,
            vel_x * np.cos(yaw), vel_x * np.sin(yaw), vel_z
        ]).reshape((6, 1))
        traj_V_G = PiecewisePolynomial.FirstOrderHold([0, t3],
                                                      np.hstack(
                                                          (vel_init, vel_fast)
                                                      ))
        t_init = t
        num_t_new = num_t + int(t3 / self.dt)
        for t_ind in range(num_t, num_t_new):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(
                station_context, traj_V_G.value(t - t_init)
            )
            if self.flag_actuate_hand:
                self.hand_position_command_port.FixValue(
                    station_context, hand_width_command
                )

            # Record
            bottle_T_all += [
                self._get_bottle_pose(plant_context).translation()
            ]

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info('Sim error!')
                info['error'] = True
                info['bottle_T_all'] = bottle_T_all
                info['bottle_T_final'] = bottle_T_all[-1]
                return np.array([]), 0, True, info
        num_t = num_t_new

        # Rest
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        vel_init = self.get_ee_vel(plant_context)
        traj_V_G = PiecewisePolynomial.FirstOrderHold([
            0, t4
        ], np.hstack((vel_init, vel_final)))
        t_init = t
        num_t_new = num_t + int(t4 / self.dt)
        for t_ind in range(num_t, num_t_new):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(
                station_context, traj_V_G.value(t - t_init)
            )
            if self.flag_actuate_hand:
                self.hand_position_command_port.FixValue(
                    station_context, hand_width_command
                )

            # Record
            bottle_T_all += [
                self._get_bottle_pose(plant_context).translation()
            ]

            # Simulate forward
            t += self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info('Sim error!')
                info['error'] = True
                info['bottle_T_all'] = bottle_T_all
                info['bottle_T_final'] = bottle_T_all[-1]
                return np.array([]), 0, True, info
        num_t = num_t_new

        # Get final bottle position
        bottle_p_final = self._get_bottle_pose(plant_context)
        bottle_T_final = bottle_p_final.translation()
        bottle_T_all += [bottle_T_final]

        # Get reward - distance to goal - no reward if bottls falls
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        dist_bottle_goal = np.linalg.norm(
            np.array(bottle_T_final[:2]) - self._goal
        )
        dist_ratio_bottle_goal = dist_bottle_goal / self.max_dist_bottle_goal
        reward = max(0, 1 - dist_ratio_bottle_goal)
        bottle_rpy_final = RollPitchYaw(bottle_p_final.rotation()).vector()
        if abs(bottle_rpy_final[0]) > self.bottle_fall_roll_threshold or abs(
            bottle_rpy_final[1]
        ) > self.bottle_fall_pitch_threshold:
            reward = 0

        # Collect info
        info['reward'] = reward  # overwrites
        info['bottle_T_all'] = bottle_T_all
        info['bottle_T_final'] = bottle_T_final

        # Always done: single step
        done = True
        return np.array([]), reward, done, info

    ##################### Getters #################

    def _get_obs(self, station_context):
        if self.use_goal:
            return self.goal.astype(np.single)  # normalized in vec_env
        else:
            return np.array([], dtype=np.single)

    def _get_bottle_vel(self, plant_context):
        return self.plant.EvalBodySpatialVelocityInWorld(
            plant_context,
            self.bottle_body,
        ).get_coeffs().reshape(6, 1)

    def _get_bottle_pose(self, plant_context):
        return self.plant.EvalBodyPoseInWorld(plant_context, self.bottle_body)
