# Modified from RobotLocomotion Group

import numpy as np
from copy import deepcopy
import logging

from pydrake.all import DiagramBuilder, RigidTransform, RotationMatrix, RollPitchYaw, ContactModel, Simulator, SpatialVelocity, InverseKinematics, Solve, Ellipsoid, Cylinder, Box, GeometryInstance, PerceptionProperties, Rgba, MakePhongIllustrationProperties, Role, RenderLabel
# from pydrake.solvers.ipopt import IpoptSolver

from adaptsim.panda.panda_station import PandaStation
from adaptsim.panda.util import set_collision_properties


class PandaEnv():

    def __init__(
        self,
        dt=0.002,
        render=False,
        use_meshcat=False,
        visualize_contact=False,
        hand_type='wsg',
        diff_ik_filter_hz=-1,
        contact_solver='sap',
        panda_joint_damping=200,
        table_type='normal',
        flag_disable_rate_limiter=False,
    ):
        self.dt = dt
        self.render = render
        self.use_meshcat = use_meshcat
        self.visualize_contact = visualize_contact
        self.panda_joint_damping = panda_joint_damping
        self.hand_type = hand_type
        if self.hand_type != 'plate':
            self.flag_actuate_hand = True
        else:
            self.flag_actuate_hand = False
        self.diff_ik_filter_hz = diff_ik_filter_hz
        self.contact_solver = contact_solver

        # Flag for whether setup is built
        self.flag_setup = False
        self.table_type = table_type
        self.flag_disable_rate_limiter = flag_disable_rate_limiter

        # Table offset
        self.table_offset = 0

        # Fixed rotation from finger to EE
        self.R_TE = RotationMatrix(RollPitchYaw(0, -np.pi, np.pi / 2))

    @property
    def q0(self):
        """
        Default joint angles. Not really used right now.
        """
        return np.array([0., -0.255, 0, -2.437, 0, 2.181, 0.785])

    def _setup(self, task=None):
        """
        Set up Panda environment in Drake. Need to load all objects here due to design of Drake.
        """
        # Load panda and Drake basics
        builder = DiagramBuilder()
        system = PandaStation(
            dt=self.dt, contact_solver=self.contact_solver,
            panda_joint_damping=self.panda_joint_damping,
            table_offset=self.table_offset, hand_type=self.hand_type
        )
        self.station = builder.AddSystem(system)
        self.hand_body = self.station.set_panda()
        # self.station.set_camera(self.camera_param)
        self.plant = self.station.get_multibody_plant()
        self.sg = self.station.get_sg()
        if self.render:
            self.visualizer = self.station.get_visualizer(
                use_meshcat=self.use_meshcat
            )
        else:
            self.visualizer = None

        # Set contact model - kPoint, kHydroelastic, kHydroelasticWithFallback
        self.plant.set_contact_model(ContactModel.kHydroelasticWithFallback)

        # Load table
        self.table_model_index, table_body_index = self.station.set_table(
            self.table_type
        )
        self.table_body = self.plant.get_body(table_body_index)

        # Load objects - child - more like a template to be modified
        self.load_objects()

        # Finalize, get controllers
        self.plant.Finalize()
        self.station.set_arm_controller(
            self.diff_ik_filter_hz, self.flag_disable_rate_limiter
        )
        if self.flag_actuate_hand:
            self.station.set_hand_controller()
        self.station.Finalize(self.visualize_contact)

        # Get controllers
        self.panda = self.station.get_panda()
        self.hand = self.station.get_hand()
        self.state_integrator = self.station.get_state_integrator()
        self.state_interpolator = self.station.get_state_interpolator()
        self.panda_controller = self.station.get_arm_controller()
        self.hand_controller = self.station.get_hand_controller()

        # Get ports
        self.contact_results_output_port = self.plant.get_contact_results_output_port(
        )
        self.reaction_forces_output_port = self.plant.get_reaction_forces_output_port(
        )
        self.table_force_output_port = self.plant.get_generalized_contact_forces_output_port(
            self.table_model_index
        )
        self.panda_force_output_port = self.plant.get_generalized_contact_forces_output_port(
            self.panda
        )
        self.panda_estimated_state_port = self.plant.get_state_output_port(
            self.panda
        )
        self.panda_desired_state_port = self.station.GetOutputPort(
            "panda_desired_state"
        )
        self.integrator_output_port = self.station.GetOutputPort(
            "integrator_output"
        )
        self.ik_result_port = self.station.GetInputPort("ik_result")
        self.V_WG_command_port = self.station.GetInputPort("V_WG_command")
        self.V_J_command_port = self.station.GetInputPort("V_J_command")
        if self.flag_actuate_hand:
            self.hand_force_measure_port = self.station.GetOutputPort(
                "hand_force_measured"
            )
            self.hand_state_measure_port = self.station.GetOutputPort(
                "hand_state_measured"
            )
            self.hand_position_command_port = self.station.GetInputPort(
                "hand_position_command"
            )

        # Set up simulator
        diagram = builder.Build()
        self._diagram_context = diagram.CreateDefaultContext()
        diagram.Publish(self._diagram_context)
        self.simulator = Simulator(diagram, self._diagram_context)
        self._diagram_context_init = self._diagram_context.Clone()
        # self._state_init = self._diagram_context.Clone().get_state()

        # Use separate plant for arm controller
        self.panda_c = self.station.get_panda_c()
        self.controller_plant = self.station.get_controller_plant()

        # Get diagram PDF
        # import pydotplus
        # pydot_graph = pydotplus.graph_from_dot_data(diagram.GetGraphvizString())
        # pydot_graph.write_pdf("panda/diagram/scoop.pdf")

    def reset(self, task=None):
        """
        Set up simulator if first time. Reset the arm and gripper.
        """
        if task is None:
            task = self.task
        else:
            self.task = task

        # Re-build or reset context
        if not self.flag_setup:
            self._setup(task)
            self.flag_setup = True
        else:
            # Set continuous/abstract/discrete states from before
            diagram_context = self.simulator.get_mutable_context()
            continuous_state = self._diagram_context_init.get_continuous_state(
            ).CopyToVector()
            diagram_context.SetContinuousState(continuous_state)
            for index in range(
                self._diagram_context_init.num_discrete_state_groups()
            ):
                discrete_state = self._diagram_context_init.get_discrete_state(
                    index
                ).CopyToVector()
                diagram_context.SetDiscreteState(index, discrete_state)
            for index in range(
                self._diagram_context_init.num_abstract_states()
            ):
                abstract_state = self._diagram_context_init.get_abstract_state(
                    index
                )
                diagram_context.SetAbstractState(index, abstract_state)
            diagram_context.SetTime(0.0)

        # Move arm away
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        self.set_arm(plant_context, self.q0)
        if self.flag_actuate_hand:
            self.set_gripper(plant_context, self.finger_init_pos)  # from child
        return task

    def reset_task(self, task=None):
        """
        Reset the task for the environment.
        """
        raise NotImplementedError

    def step(self):
        """
        Apply action, move robot, get observation, calculate reward, check if done.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Set the seed of the environment. No torch rn.
        """
        self.rng = np.random.default_rng(seed=seed)
        return [seed]

    def ik(
        self,
        plant_context,
        controller_plant_context,
        frame='arm',
        p_e=None,
        T_e=None,
        R_e=None,
    ):
        if p_e is not None:
            T_e = p_e.translation()
            R_e = p_e.rotation()
        ik = InverseKinematics(
            self.controller_plant,
            controller_plant_context,
            with_joint_limits=True,  # much faster
        )
        q_cur = self.plant.GetPositions(
            plant_context
        )[:7
         ]  # use normal plant to get arm joint angles, since the control plant does not update them
        if frame == 'arm':
            ee_frame = self.controller_plant.GetFrameByName(
                "panda_link8", self.panda_c
            )
        elif frame == 'tip':
            ee_frame = self.controller_plant.GetFrameByName(
                "fingertip_frame", self.panda_c
            )
        else:
            raise 'Unknown frame error!'
        ik.AddPositionConstraint(
            ee_frame, [0, 0, 0], self.plant.world_frame(), T_e, T_e
        )
        ik.AddOrientationConstraint(
            ee_frame, RotationMatrix(), self.plant.world_frame(), R_e, 0.0
        )
        prog = ik.get_mutable_prog()
        # prog.SetSolverOption(IpoptSolver().solver_id(),
        #                      "max_iter", 500)    # speed up
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q_cur, q)
        prog.SetInitialGuess(q, q_cur)
        result = Solve(ik.prog())  # sim returns error if this line is called
        if not result.is_success():
            logging.error("IK failed")
        # print('solver is: ', result.get_solver_id().name())
        qstar_raw = result.GetSolution(q)
        qstar = qstar_raw[:7]
        # print('IK solution: ', repr(qstar))
        return qstar

    ########################## Helper ##########################

    def get_ee_pose(self, plant_context):
        return self.plant.EvalBodyPoseInWorld(
            plant_context, self.plant.GetBodyByName("panda_link8")
        )

    def get_ee_vel(self, plant_context):
        return self.plant.EvalBodySpatialVelocityInWorld(
            plant_context, self.plant.GetBodyByName("panda_link8")
        ).get_coeffs().reshape(6, 1)

    def get_joint_angles(self, plant_context):
        return self.plant.GetPositions(plant_context)[:7]

    def get_joint_velocities(self, plant_context):
        return self.plant.GetVelocities(plant_context)[:7]

    def get_ee_force_torque(self, plant_context):
        ee_ft = self.reaction_forces_output_port.Eval(plant_context)[7]
        return ee_ft.translational(), ee_ft.rotational()

    def set_arm(self, plant_context, q, qdot=[0] * 7):
        self.plant.SetPositions(plant_context, self.panda, q)
        self.plant.SetVelocities(plant_context, self.panda, qdot)

    def set_gripper(self, plant_context, g, gdot=[0] * 2):
        self.plant.SetPositions(plant_context, self.hand, [-g, g])
        self.plant.SetVelocities(plant_context, self.hand, gdot)

    def set_body_pose(self, body, plant_context, p, rpy=[0, 0, 0], rm=None):
        if rm is None:
            rm = RotationMatrix(RollPitchYaw(rpy[0], rpy[1], rpy[2]))
        self.plant.SetFreeBodyPose(plant_context, body, RigidTransform(rm, p))

    def set_body_vel(self, body, plant_context, w=[0, 0, 0], v=[0, 0, 0]):
        self.plant.SetFreeBodySpatialVelocity(
            body, SpatialVelocity(w=w, v=v), plant_context
        )

    def reset_state(self, plant_context, context, set_integrator=True):
        state_integrator_context = self.state_integrator.GetMyMutableContextFromRoot(
            context
        )
        state_interpolator_context = self.state_interpolator.GetMyMutableContextFromRoot(
            context
        )
        q = self.get_joint_angles(plant_context)
        dq = self.get_joint_velocities(plant_context)
        self.state_interpolator.set_initial_state(
            state_interpolator_context, q, dq
        )  # avoid large derivatives at the beginning
        if set_integrator:  # do not set it when using ik result instead of diff ik
            self.state_integrator.set_integral_value(
                state_integrator_context, q
            )

    def disable_diff_ik(self, context):
        """By disabling state integrator"""
        state_integrator_context = self.state_integrator.GetMyMutableContextFromRoot(
            context
        )
        self.state_integrator.set_integral_value(
            state_integrator_context, np.zeros((7))
        )

    def set_obj_dynamics(
        self,
        context_inspector,
        context,
        body,
        mu=None,
        hc_dissipation=None,
        hydro_resolution=None,
        hydro_modulus=None,
        sap_dissipation=None,
        compliance_type=None,
        flag_all_geometry=False,
    ):
        '''
        Only dynamic friction is used in point contact (static ignored). Assign role with context.
        '''
        geometry_ids = self.plant.GetCollisionGeometriesForBody(body)
        if not flag_all_geometry:
            geometry_ids = geometry_ids[0:1]
        for geometry_id in geometry_ids:
            set_collision_properties(
                self.sg,
                context,
                context_inspector,
                self.plant,
                geometry_id,
                hc_dissipation=hc_dissipation,
                sap_dissipation=sap_dissipation,
                static_friction=mu,
                dynamic_friction=mu,
                hydro_resolution=hydro_resolution,
                hydro_modulus=hydro_modulus,
                compliance_type=compliance_type,
            )

    def replace_body(
        self,
        context,
        context_inspector,
        body,
        frame_id,
        geom_type,
        x_dim,
        y_dim,
        z_dim,
        x=0,
        y=0,
        z=0,
        roll=0,
        pitch=0,
        yaw=0,
        color=[0.659, 0.839, 0.514, 1],
        render_label=100,
        visual_name='visual',
        collision_name='collision',
    ):
        """Replace body at frame.
        """
        # Remove old geometries if exists - save a copy of old proximity properties - probably not needed since changing obj dynamics later
        source_id = self.plant.get_source_id()
        old_visual_id = context_inspector.GetGeometries(
            frame_id, Role.kPerception
        )
        assert len(old_visual_id) <= 1
        illu_properties = None
        if len(old_visual_id) == 1:
            old_visual_id = old_visual_id[0]
            perc_properties = deepcopy(
                context_inspector.GetPerceptionProperties(old_visual_id)
            )
            illu_properties = deepcopy(
                context_inspector.GetIllustrationProperties(old_visual_id)
            )
            # print(context_inspector.GetIllustrationProperties(old_visual_id))
            # print(context_inspector.GetPerceptionProperties(old_visual_id))
            self.sg.RemoveGeometry(
                context=context, source_id=source_id, geometry_id=old_visual_id
            )
        old_prox_id = context_inspector.GetGeometries(
            frame_id, Role.kProximity
        )
        assert len(old_prox_id) <= 1
        prox_properties = None
        if len(old_prox_id) == 1:
            old_prox_id = old_prox_id[0]
            prox_properties = deepcopy(
                context_inspector.GetProximityProperties(old_prox_id)
            )  # no illustration or perception properties
            self.sg.RemoveGeometry(
                context=context, source_id=source_id, geometry_id=old_prox_id
            )

        # Determine geometry/shape type - do not keep creating if no link speficied
        if geom_type == 'ellipsoid':
            visual_shape = Ellipsoid(x_dim, y_dim, z_dim)
        elif geom_type == 'cylinder':
            visual_shape = Cylinder(x_dim, z_dim * 2)
        elif geom_type == 'box':
            visual_shape = Box(x_dim * 2, y_dim * 2, z_dim * 2)
        else:
            return 0
        prox_shape = deepcopy(visual_shape)
        transform = RigidTransform(RollPitchYaw(roll, pitch, yaw), [x, y, z])

        # Create new visual geometry with illustration and perception properties
        new_visual = GeometryInstance(
            X_PG=transform, shape=visual_shape, name=visual_name
        )
        if illu_properties is None:
            illu_properties = MakePhongIllustrationProperties(color)
        new_visual.set_illustration_properties(illu_properties)
        if perc_properties is None:
            perc_properties = PerceptionProperties()
            perc_properties.AddProperty(
                "phong", "diffuse",
                Rgba(color[0], color[1], color[2], color[3])
            )
            perc_properties.AddProperty(
                "label", "id", RenderLabel(render_label)
            )
        new_visual.set_perception_properties(perc_properties)

        # Create new proximity geometry
        new_prox = GeometryInstance(
            X_PG=transform, shape=prox_shape, name=collision_name
        )
        new_prox.set_proximity_properties(prox_properties)

        # Register new geometry on SG
        new_visual_id = self.sg.RegisterGeometry(
            context=context, source_id=source_id, frame_id=frame_id,
            geometry=new_visual
        )
        new_prox_id = self.sg.RegisterGeometry(
            context=context, source_id=source_id, frame_id=frame_id,
            geometry=new_prox
        )

        # Swap geometry on MbP
        self.plant.SwapCollisionGeometries(body, old_prox_id, new_prox_id)
        return 1
