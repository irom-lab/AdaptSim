import numpy as np
import scipy.interpolate
import logging
from pydrake.all import RotationMatrix, RollPitchYaw, RigidTransform, PiecewisePolynomial

from adaptsim.env.scoop_env import ScoopEnv
from adaptsim.panda.util import AddMultibodyTriad


class ScoopFullEnv(ScoopEnv):

    def __init__(
        self,
        dt=0.002,
        render=False,
        visualize_contact=False,
        hand_type='panda_foam',
        diff_ik_filter_hz=500,
        contact_solver='sap',
        panda_joint_damping=200,
        table_type='normal',
        flag_disable_rate_limiter=False,
        num_link=1,
        veggie_x=0.68,
        traj_type='',
    ):
        super(ScoopFullEnv, self).__init__(
            dt=dt,
            render=render,
            visualize_contact=visualize_contact,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
            table_type=table_type,
            flag_disable_rate_limiter=flag_disable_rate_limiter,
            num_link=num_link,
            veggie_x=veggie_x,
            traj_type=traj_type,
        )

    def load_objects(self,):
        super().load_objects()

        # Load spatula holder
        holder_path = 'asset/spatula_holder/spatula_holder.sdf'
        self.holder_model_index, self.holder_body_index = \
            self.station.AddModelFromFile(holder_path, name='spatula_holder')
        self.holder_base = self.plant.get_body(self.holder_body_index[0])

        # Get useful frames of the spatula
        self.spatula_holder_tip_align_blade_frame = self.plant.GetFrameByName(
            "spatula_holder_tip_align_blade", self.holder_model_index
        )
        AddMultibodyTriad(
            self.spatula_holder_tip_align_blade_frame,
            self.sg,
            length=.02,
            radius=0.001,
        )

        # Weld spatula holder to table
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("spatula_holder_origin"),
            RigidTransform([0.5, -0.2, self.table_offset])
        )

        # Load bowl
        # self.bowl_model_index, self.bowl_body_index = \
        #     self.station.AddModelFromFile(bowl_path, name='bowl')
        # self.bowl_base = self.plant.get_body(self.bowl_body_index[0])

    def reset(self, task=None):
        """
        Call parent to reset arm and gripper positions (build if first-time). Reset veggies and task. Do not initialize simulator.
        """
        obs = super().reset(task)

        # Get new context
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        context_inspector = query_object.inspector()

        # Get transform from origin to holder tip
        self.p_holder_tip = self.spatula_holder_tip_align_blade_frame.CalcPose(
            plant_context, self.plant.world_frame()
        )

        return obs

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
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        inspector = query_object.inspector()

        # Extract action
        s_yaw = 0
        s_pitch_init = action[0]
        s_x_veggie_tip = -0.02
        xd_1, pd_1, pd_2 = action[1:]

        # Place bowl
        self.bowl_loc = [0.70, 0.20, self.table_offset]
        # self.set_body_pose(self.bowl_base, plant_context, p=self.bowl_loc)

        # Put spatula on holder
        p_spatula_base = self.p_holder_tip.multiply(
            self.p_spatula_tip_to_spatula_base
        )
        self.set_body_pose(
            self.spatula_base, plant_context,
            p=p_spatula_base.translation() + np.array([0, 0, 0.002]),
            rm=p_spatula_base.rotation()
        )

        # Initialize state interpolator/integrator
        self.reset_state(plant_context, context)

        # Reset simulation
        sim_context = self.simulator.get_mutable_context()
        sim_context.SetTime(0.)
        self.simulator.Initialize()

        ######################## Trajectory ########################
        t_ind_init = 1
        t_ind_final = 1

        ###########################################################
        ################# Let spatula reset #######################
        ###########################################################
        t_seg = 0.5
        t_ind_init = t_ind_final
        t_ind_final += int(t_seg / self.dt)
        hand_width_command = 0.2
        for t_ind in range(t_ind_init, t_ind_final):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)

            # For some reason, using ik initially makes robot jump slightly
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, np.zeros((6)))
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}
            # time.sleep(0.2)

        ###########################################################
        ####################### Get poses #########################
        ###########################################################

        # Gripper pose when grasping the spatula - no yaw yet
        if self.hand_type == 'panda_foam':
            T_grip_to_ee = np.array([0, 0, 0.16])
        elif self.hand_type == 'panda':
            T_grip_to_ee = np.array([0, 0, 0.10])
        else:
            raise "Unknown hand type!"

        # grasp the spatula such that when it is moved to the table, with the pitch angle, the EE is stright down
        p_spatula_tip = self.spatula_tip_frame.CalcPoseInWorld(plant_context)
        R_grip_align = RotationMatrix(RollPitchYaw(0, -s_pitch_init, 0))  #!
        R_grip_pre_grasp = p_spatula_tip.rotation()
        R_grip_pre_grasp = R_grip_pre_grasp.multiply(R_grip_align)  #!
        T_grip_pre_grasp = self.spatula_grasp_frame.CalcPoseInWorld(
            plant_context
        ).translation()
        p_grip_pre_grasp = RigidTransform(R_grip_pre_grasp, T_grip_pre_grasp)

        # grip to EE
        p_grip_to_ee = RigidTransform(self.R_TE, T_grip_to_ee)
        p_ee_pre_grasp = p_grip_pre_grasp.multiply(p_grip_to_ee)
        q_pre_grasp = self.ik(
            plant_context, controller_plant_context, p_e=p_ee_pre_grasp
        )
        print(
            'q_pre_grasp = ',
            np.array2string(
                q_pre_grasp, precision=3, separator=',', suppress_small=True
            )
        )
        # Extra pose: waypoint before reaching down
        p_ee_pre_grasp_top = p_ee_pre_grasp.multiply(
            RigidTransform(p=[0, 0, -0.1])
        )  # above the grasp to avoid contacting the spatula
        q_pre_grasp_top = self.ik(
            plant_context, controller_plant_context, p_e=p_ee_pre_grasp_top
        )
        print(
            'q_pre_grasp_top = ',
            np.array2string(
                q_pre_grasp_top, precision=3, separator=',',
                suppress_small=True
            )
        )

        # Extra pose: waypoint after grasping - move back a bit to avoid sliding along the holder tip
        p_ee_pre_grasp_back = p_ee_pre_grasp.multiply(
            RigidTransform(p=[0, 0.03, -0.05])
        )  # above the grasp to avoid contacting the spatula
        q_pre_grasp_back = self.ik(
            plant_context, controller_plant_context, p_e=p_ee_pre_grasp_back
        )
        print(
            'q_pre_grasp_back = ',
            np.array2string(
                q_pre_grasp_back, precision=3, separator=',',
                suppress_small=True
            )
        )

        ###########################################################
        ############## Move to pre_grasp at holder ################
        ###########################################################
        t_seg = 4
        t_ind_init = t_ind_final
        t_ind_final += int(t_seg / self.dt)  # 0.2
        hand_width_command = 0.2
        t_init = t
        traj_ik = PiecewisePolynomial.FirstOrderHold(
            [0, t_seg - 1, t_seg],
            np.vstack((self.q0, q_pre_grasp_top, q_pre_grasp)).T
        )
        for t_ind in range(t_ind_init, t_ind_final):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)

            self.disable_diff_ik(context)
            self.ik_result_port.FixValue(
                station_context, traj_ik.value(t - t_init)
            )
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}

        # Close gripper
        t_seg = 0.5
        t_ind_init = t_ind_final
        t_ind_final += int(t_seg / self.dt)
        hand_width_command = 0
        for t_ind in range(t_ind_init, t_ind_final):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)
            # hand_position_measure = self.hand_state_measure_port.Eval(station_context)
            # hand_force_measure = self.hand_force_measure_port.Eval(station_context)

            self.disable_diff_ik(context)
            self.ik_result_port.FixValue(station_context, q_pre_grasp)
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}

        ###########################################################
        ####################### Get poses #########################
        ###########################################################

        # Re-calculate tip frame
        p_spatula_tip = self.spatula_tip_frame.CalcPoseInWorld(plant_context)

        # Relative pose between tip and EE - this is fixed after grasp
        p_spatula_tip_to_ee = p_spatula_tip.InvertAndCompose(
            self.get_ee_pose(plant_context)
        )

        # Veggie to spatuala tip - account for pitch
        p_veggie_to_spatula_tip = RigidTransform(
            RollPitchYaw(0, s_pitch_init, s_yaw), [s_x_veggie_tip, 0., 0.]
        )
        p_spatula_tip_pre_scoop = self.veggie_fixed_frame.CalcPoseInWorld(
            plant_context
        ).multiply(p_veggie_to_spatula_tip)
        p_spatula_tip_align = RigidTransform(
            rpy=RollPitchYaw(0, 0, 0), p=[0, 0, -0.002]
        )
        p_spatula_tip_pre_scoop = p_spatula_tip_pre_scoop.multiply(
            p_spatula_tip_align
        )
        p_ee_pre_scoop = p_spatula_tip_pre_scoop.multiply(p_spatula_tip_to_ee)
        q_pre_scoop = self.ik(
            plant_context, controller_plant_context, p_e=p_ee_pre_scoop
        )

        # Extra pose: waypoint above pre_scoop
        p_ee_pre_scoop_up = p_ee_pre_scoop.multiply(
            RigidTransform(p=[0, 0, -0.02])
        )  # above the grasp to avoid contacting the spatula
        q_pre_scoop_up = self.ik(
            plant_context, controller_plant_context, p_e=p_ee_pre_scoop_up
        )
        print(
            'q_pre_scoop_up = ',
            np.array2string(
                q_pre_scoop_up, precision=3, separator=',', suppress_small=True
            )
        )
        print(
            'q_pre_scoop = ',
            np.array2string(
                q_pre_scoop, precision=3, separator=',', suppress_small=True
            )
        )

        ###########################################################
        ################### Move to pre scoop #####################
        ###########################################################

        t_seg = 4
        t_init = t
        t_ind_init = t_ind_final
        t_ind_final += int(t_seg / self.dt)
        traj_ik = PiecewisePolynomial.FirstOrderHold(
            [0, 0.5, t_seg - 1, t_seg],
            np.vstack(
                (q_pre_grasp, q_pre_grasp_back, q_pre_scoop_up, q_pre_scoop)
            ).T
        )
        hand_width_command = -0.5
        for t_ind in range(t_ind_init, t_ind_final):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)

            self.disable_diff_ik(context)
            self.ik_result_port.FixValue(
                station_context, traj_ik.value(t - t_init)
            )
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # p_spatula_tip = self.spatula_tip_frame.CalcPoseInWorld(plant_context)
            # p_spatula_tip_to_ee = p_spatula_tip.InvertAndCompose(self.get_ee_pose(plant_context))
            # print(p_spatula_tip_to_ee.rotation().ToAngleAxis())
            # print(p_spatula_tip_to_ee.translation())
            # t_all += [p_spatula_tip_to_ee.translation()]

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}

        ###########################################################
        ################### Scoop with Diff IK ####################
        ###########################################################

        # Initialize state interpolator/integrator
        self.reset_state(plant_context, context)

        # Time for spline
        num_t_init = t_ind
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
        zd = [0, -0.01, 0.02, 0.02, 0, 0]
        poly_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
        zds = poly_zd(ts)

        # Go through trajectory
        num_t = len(ts)
        for t_ind in range(0, num_t):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)

            V_G = np.array([0, pitchds[t_ind], 0, xds[t_ind], 0,
                            zds[t_ind]]).reshape(6, 1)
            self.ik_result_port.FixValue(station_context, np.zeros((7)))
            self.V_WG_command_port.FixValue(station_context, V_G)
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Simulate forward
            t = (t_ind+num_t_init) * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}
            # time.sleep(0.01)

        ###########################################################
        ################### IK to move to bowl ####################
        ###########################################################
        # Solve IK for bowl location - make spatula level
        T_ee_pre_dump = np.array(self.bowl_loc
                                ) + np.array([-0.13, -0.01, 0.28])
        q_pre_dump = self.ik(
            plant_context, controller_plant_context, T_e=T_ee_pre_dump,
            R_e=p_spatula_tip_to_ee.rotation()
        )  # since spatula tip is aligned with world when it is level
        T_ee_dump = T_ee_pre_dump + np.array([0, 0.05, -0.03])
        q_dump = self.ik(
            plant_context, controller_plant_context, T_e=T_ee_dump,
            R_e=p_spatula_tip_to_ee.rotation().multiply(
                RotationMatrix(RollPitchYaw(0, np.pi / 4, 0))
            )
        )
        print(
            'q_pre_dump = ',
            np.array2string(
                q_pre_dump, precision=3, separator=',', suppress_small=True
            )
        )
        print(
            'q_dump = ',
            np.array2string(
                q_dump, precision=3, separator=',', suppress_small=True
            )
        )

        t_seg = 3
        t_init = t
        t_ind_init = int(t / self.dt)
        t_ind_final = t_ind_init + int(t_seg / self.dt)
        traj_ik = PiecewisePolynomial.FirstOrderHold(
            [0, t_seg - 1, t_seg],
            np.vstack(
                (self.get_joint_angles(plant_context), q_pre_dump, q_dump)
            ).T
        )
        hand_width_command = -0.5
        for t_ind in range(t_ind_init, t_ind_final):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)

            self.disable_diff_ik(context)
            self.ik_result_port.FixValue(
                station_context, traj_ik.value(t - t_init)
            )
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}

        ###########################################################
        ################# IK to put spatula back ##################
        ###########################################################
        offset = [
            0, 0.002, 0
        ]  # move back slightly more to avoid spatula tip hitting the holder tip, yikes
        p_ee_pre_grasp_final = p_ee_pre_grasp.multiply(
            RigidTransform(p=offset)
        )
        q_pre_grasp_final = self.ik(
            plant_context, controller_plant_context, p_e=p_ee_pre_grasp_final
        )
        p_ee_pre_grasp_back_final = p_ee_pre_grasp_back.multiply(
            RigidTransform(p=offset)
        )  # move back slightly more, yikes
        q_pre_grasp_back_final = self.ik(
            plant_context, controller_plant_context,
            p_e=p_ee_pre_grasp_back_final
        )
        print(
            'q_pre_grasp_back_final = ',
            np.array2string(
                q_pre_grasp_back_final, precision=3, separator=',',
                suppress_small=True
            )
        )
        print(
            'q_pre_grasp_final = ',
            np.array2string(
                q_pre_grasp_final, precision=3, separator=',',
                suppress_small=True
            )
        )

        t_seg = 5
        t_init = t
        t_ind_init = t_ind_final
        t_ind_final += int(t_seg / self.dt)
        traj_ik = PiecewisePolynomial.FirstOrderHold(
            [0, t_seg - 1, t_seg],
            np.vstack((q_dump, q_pre_grasp_back_final, q_pre_grasp_final)).T
        )
        hand_width_command = -0.5
        for t_ind in range(t_ind_init, t_ind_final):
            context = self.simulator.get_mutable_context()
            plant_context = self.plant.GetMyContextFromRoot(context)

            self.disable_diff_ik(context)
            self.ik_result_port.FixValue(
                station_context, traj_ik.value(t - t_init)
            )
            self.hand_position_command_port.FixValue(
                station_context, hand_width_command
            )

            # Simulate forward
            t = t_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(t)
            except RuntimeError as e:
                logging.info(f'Sim error at time {t}!')
                return self._get_obs(), 0, True, {}

        # Get info
        info = {}

        # Get reward - veggie piece on spatula
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        # contact_results_list = self.contact_results_output_port.Eval(plant_context)
        pose_blade = self.spatula_blade_frame.CalcPoseInWorld(plant_context)
        reward = 0
        for veggie_body in self.veggie_base_all:
            veggie_pose = self.plant.EvalBodyPoseInWorld(
                plant_context, veggie_body
            )
            blade_veggie = pose_blade.InvertAndCompose(veggie_pose)
            # veggie_vel = self.plant.EvalBodySpatialVelocityInWorld(plant_context, veggie_body).translational()
            if blade_veggie.translation()[2] > 0.001:
                reward += 1 / self.task.obj_num
        info['reward'] = reward

        # Always done: single step
        done = True
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        return self._get_obs(plant_context), reward, done, info
