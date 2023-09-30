import os
import numpy as np
import pydrake
from pydrake.all import (
    LoadModelDirectives, ProcessModelDirectives, GeometrySet, Parser,
    DiagramBuilder, Diagram, MultibodyPlant, Demultiplexer,
    InverseDynamicsController, Adder, PassThrough,
    StateInterpolatorWithDiscreteDerivative, RigidTransform, RollPitchYaw,
    Integrator, SchunkWsgPositionController,
    MakeMultibodyStateToWsgStateSystem, DrakeVisualizer,
    ConnectContactResultsToDrakeVisualizer, FixedOffsetFrame,
    MultibodyPlantConfig, AddMultibodyPlant, DiscreteContactSolver
)
from adaptsim.panda.differential_ik import PseudoInverseController
from adaptsim.panda.discrete_filter import DiscreteLowPassFilter
from adaptsim.panda.rate_limiter import RateLimiter
from adaptsim.panda.scenarios import AddPanda, AddHand, AddRgbdSensor
from adaptsim.panda.util import AddMultibodyTriad


def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def AddPackagePaths(parser):
    parser.package_map().PopulateFromFolder(FindResource(""))
    parser.package_map().Add(
        "manipulation_station",
        os.path.join(
            pydrake.common.GetDrakePath(),
            "examples/manipulation_station/models"
        )
    )
    parser.package_map().Add("drake_models", "asset/")


def deg_to_rad(deg):
    return deg * np.pi / 180.0


class PandaStation(Diagram):

    def __init__(
        self,
        dt=0.002,
        contact_solver='sap',
        panda_joint_damping=200,
        table_offset=0,
        hand_type=None,
    ):
        Diagram.__init__(self)
        self.dt = dt
        self.panda_joint_damping = panda_joint_damping
        self.table_offset = table_offset
        self.hand_type = hand_type

        # Initialie builder and plant
        self.builder = DiagramBuilder()
        plant_cfg = MultibodyPlantConfig(
            time_step=dt, discrete_contact_solver=contact_solver
        )
        self.plant, self.sg = AddMultibodyPlant(plant_cfg, self.builder)
        self.plant.set_name("plant")

        self.set_name("panda_station")
        self.object_ids = []
        self.object_poses = []
        self.camera_info = {}  # dictionary in the form name: pose
        self.body_info = []  # path: (name, body_index)

        # Controller plant
        self.controller_plant = self.builder.AddSystem(
            MultibodyPlant(time_step=dt)
        )
        if contact_solver == 'sap':
            self.controller_plant.set_discrete_contact_solver(
                DiscreteContactSolver.kSap
            )
        else:
            self.controller_plant.set_discrete_contact_solver(
                DiscreteContactSolver.kTamsi
            )

    def set_table(self, table_type=True):
        if table_type == 'overlap':
            directive = 'asset/table/table_top_overlap.yaml'
        elif table_type == 'normal':
            directive = 'asset/table/table_top.yaml'
        parser = Parser(self.plant)
        AddPackagePaths(parser)
        table_info = ProcessModelDirectives(
            LoadModelDirectives(directive), self.plant, parser
        )
        table_model_index = table_info[0].model_instance
        table_body_index = self.plant.GetBodyIndices(table_model_index)[0]

        return table_model_index, table_body_index

    def set_panda(self,):
        self.panda = AddPanda(
            self.plant, joint_damping=self.panda_joint_damping
        )  # no hand inertia
        self.hand, hand_body = AddHand(
            self.plant,
            panda_model_instance=self.panda,
            # welded=False,
            type=self.hand_type
        )

        # Set joint acceleration limits - position and velocity limits already specified in SDF
        acc_limit = [15, 7.5, 10, 12.5, 15, 20, 20]
        for joint_ind in range(1, 8):
            joint_name = 'panda_joint' + str(joint_ind)
            joint = self.plant.GetJointByName(joint_name, self.panda)
            joint.set_acceleration_limits([-acc_limit[joint_ind - 1]],
                                          [acc_limit[joint_ind - 1]])
        return hand_body

    def set_camera(self, camera_param):
        if camera_param is None:
            return

        # Add box
        X_Camera = RigidTransform(
            RollPitchYaw(camera_param.euler).ToRotationMatrix(),
            np.array(camera_param.pos) + np.array([0, 0, self.table_offset])
        )  # 0, -pi, pi/2; 0.7, 0, 0.3
        parser = Parser(self.plant)
        camera_instance = parser.AddModelFromFile(
            'asset/camera/camera_box.sdf'
        )
        camera_frame = self.plant.GetFrameByName("base", camera_instance)
        self.plant.WeldFrames(self.plant.world_frame(), camera_frame, X_Camera)
        AddMultibodyTriad(
            camera_frame,
            self.sg,
            length=.1,
            radius=0.005,
        )

        # Add camera and export outputs
        camera, self.renderer = AddRgbdSensor(
            self.builder, self.sg, X_PC=RigidTransform(),
            parent_frame_id=self.plant.GetBodyFrameIdOrThrow(
                camera_frame.body().index()
            ), camera_param=camera_param
        )
        camera.set_name("rgbd_sensor")
        self.builder.ExportOutput(
            camera.color_image_output_port(),
            "color_image",
        )
        self.builder.ExportOutput(
            camera.depth_image_32F_output_port(),
            "depth_image",
        )
        self.builder.ExportOutput(
            camera.label_image_output_port(),
            "label_image",
        )

    def get_visualizer(self, use_meshcat=False):
        if use_meshcat:
            from pydrake.all import MeshcatVisualizerCpp
            from util.meshcat_cpp_utils import StartMeshcat
            meshcat = StartMeshcat()
            visualizer = MeshcatVisualizerCpp.AddToBuilder(
                self.builder,
                self.sg.get_query_output_port(),
                meshcat,
            )
        else:
            visualizer = DrakeVisualizer.AddToBuilder(
                builder=self.builder,
                scene_graph=self.sg,
            )
        return visualizer

    def fix_collisions(self):
        # fix collisions in this model by removing collisions between
        # panda_link5<->panda_link7 and panda_link7<->panda_hand
        # panda_link5 =  self.plant.GetFrameByName("panda_link5").body()
        # panda_link5 =  GeometrySet(
        #         self.plant.GetCollisionGeometriesForBody(panda_link5))
        panda_link7 = self.plant.GetFrameByName("panda_link7").body()
        panda_link7 = GeometrySet(
            self.plant.GetCollisionGeometriesForBody(panda_link7)
        )
        panda_hand = self.plant.GetFrameByName("panda_hand").body()
        panda_hand = GeometrySet(
            self.plant.GetCollisionGeometriesForBody(panda_hand)
        )
        # self.sg.ExcludeCollisionsBetween(panda_link5, panda_link7)
        self.collision_filter_manager = self.sg.collision_filter_manager()
        # self.collision_filter_manager.Apply(CollisionFilterDeclaration.ExcludeBetween(panda_link7, panda_hand))

    def Finalize(self, visualize_contact):
        assert self.panda.is_valid(), "No panda model added"
        assert self.hand.is_valid(), "No panda hand model added"

        # Export state ports
        self.builder.ExportOutput(
            self.sg.get_query_output_port(), "geometry_query"
        )
        self.builder.ExportOutput(
            self.plant.get_contact_results_output_port(), "contact_results"
        )
        self.builder.ExportOutput(
            self.plant.get_state_output_port(), "plant_continuous_state"
        )

        # Send contact info to visualizer
        if visualize_contact:
            ConnectContactResultsToDrakeVisualizer(
                builder=self.builder,
                plant=self.plant,
                scene_graph=self.sg,
            )

        # For visualization
        self.builder.ExportOutput(
            self.sg.get_query_output_port(), "query_object"
        )
        self.builder.BuildInto(self)

    def set_arm_controller(
        self, diff_ik_filter_hz, flag_disable_rate_limiter=False
    ):
        # Export panda joint state outputs with demux
        num_panda_positions = self.plant.num_positions(self.panda)
        demux = self.builder.AddSystem(
            Demultiplexer(2 * num_panda_positions, num_panda_positions)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.panda),
            demux.get_input_port()
        )

        # Initialize plant for the controller
        self.panda_c = AddPanda(
            self.controller_plant, joint_damping=self.panda_joint_damping,
            hand_type=self.hand_type
        )

        # Set joint acceleration limits - position and velocity limits already specified in SDF
        acc_limit = [15, 7.5, 10, 12.5, 15, 20, 20]
        for joint_ind in range(1, 8):
            joint_name = 'panda_joint' + str(joint_ind)
            joint = self.controller_plant.GetJointByName(
                joint_name, self.panda_c
            )
            joint.set_acceleration_limits([-acc_limit[joint_ind - 1]],
                                          [acc_limit[joint_ind - 1]])

        # Add fixed frame for the point between fingers (for ik), to arm (not hand!)
        self.fingertip_frame = self.controller_plant.AddFrame(
            FixedOffsetFrame(
                "fingertip_frame",
                self.controller_plant.GetFrameByName(
                    "panda_link8", self.panda_c
                ), RigidTransform([0, 0, 0.16])
            )
        )
        self.controller_plant.Finalize()

        # Add arm controller
        kp = np.array([2000, 2000, 2000, 2000, 2000, 2000, 2000]) * 1
        kd = 2 * np.sqrt(kp)
        ki = np.ones(7)
        self.panda_controller = self.builder.AddSystem(
            InverseDynamicsController(
                self.controller_plant, kp=kp, kd=kd, ki=ki,
                has_reference_acceleration=False
            )
        )
        self.panda_controller.set_name("panda_controller")
        self.builder.Connect(
            self.plant.get_state_output_port(self.panda),
            self.panda_controller.get_input_port_estimated_state()
        )

        # Add FF torque
        adder = self.builder.AddSystem(Adder(2, num_panda_positions))
        self.builder.Connect(
            self.panda_controller.get_output_port_control(),
            adder.get_input_port(0)
        )
        torque_passthrough = self.builder.AddSystem(
            PassThrough([0] * num_panda_positions)
        )
        self.builder.Connect(
            torque_passthrough.get_output_port(), adder.get_input_port(1)
        )
        self.builder.ExportInput(
            torque_passthrough.get_input_port(), "panda_feedforward_torque"
        )
        self.builder.Connect(
            adder.get_output_port(),
            self.plant.get_actuation_input_port(self.panda)
        )
        self.builder.Connect(
            adder.get_output_port(),
            self.controller_plant.get_actuation_input_port(self.panda_c)
        )

        # Add Differential IK
        diff_ik = self.builder.AddSystem(
            PseudoInverseController(self.controller_plant, self.dt)
        )
        diff_ik.set_name("PseudoInverseController")

        # Connnect joint measurements to diff ik from demux
        self.builder.Connect(
            demux.get_output_port(0), diff_ik.GetInputPort("q")
        )
        self.builder.Connect(
            demux.get_output_port(1), diff_ik.GetInputPort("qdot")
        )

        # Connect velocity command to diff ik
        V_WG_command = self.builder.AddSystem(PassThrough(6))
        self.builder.ExportInput(V_WG_command.get_input_port(), "V_WG_command")
        self.builder.Connect(
            V_WG_command.get_output_port(), diff_ik.GetInputPort("V_WG")
        )

        # Optional: pass diff ik output through low pass filter
        if diff_ik_filter_hz > 0:
            lp_filter = self.builder.AddSystem(
                DiscreteLowPassFilter(self.dt, diff_ik_filter_hz)
            )
            self.builder.Connect(
                diff_ik.get_output_port(),
                lp_filter.get_input_port(0),
            )
            diff_ik_output_port = lp_filter.get_output_port(0)
        else:
            diff_ik_output_port = diff_ik.get_output_port()

        # Add diff ik output and v_joint input
        adder_diffik_vj = self.builder.AddSystem(Adder(2, num_panda_positions))
        self.builder.Connect(
            diff_ik_output_port, adder_diffik_vj.get_input_port(0)
        )
        v_j_input = self.builder.AddSystem(PassThrough(num_panda_positions))
        self.builder.Connect(
            v_j_input.get_output_port(), adder_diffik_vj.get_input_port(1)
        )
        self.builder.ExportInput(v_j_input.get_input_port(), "V_J_command")

        # Rate limiter
        rate_limiter = self.builder.AddSystem(
            RateLimiter(self.dt, flag_disable_rate_limiter)
        )
        rate_limiter.set_name("RateLimiter")
        self.builder.Connect(
            adder_diffik_vj.get_output_port(),
            rate_limiter.GetInputPort('qdot_d')
        )
        self.builder.Connect(
            demux.get_output_port(1), rate_limiter.GetInputPort("qdot")
        )

        # Feed to integrator
        self.state_integrator = self.builder.AddSystem(Integrator(7))
        self.state_integrator.set_name("state_integrator")
        self.builder.Connect(
            rate_limiter.get_output_port(),
            self.state_integrator.get_input_port()
        )
        self.builder.ExportOutput(
            self.state_integrator.get_output_port(), "integrator_output"
        )

        # Add integrator output and IK input
        ik_input = self.builder.AddSystem(PassThrough(num_panda_positions))
        self.builder.ExportInput(ik_input.get_input_port(), "ik_result")
        adder_integrator_ik = self.builder.AddSystem(
            Adder(2, num_panda_positions)
        )
        self.builder.Connect(
            self.state_integrator.get_output_port(),
            adder_integrator_ik.get_input_port(0)
        )
        self.builder.Connect(
            ik_input.get_output_port(), adder_integrator_ik.get_input_port(1)
        )

        # Add interpolator to find velocity command based on positional commands, between DIK and IDC
        self.state_interpolator = self.builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                num_panda_positions, self.dt, suppress_initial_transient=False
            )
        )
        self.state_interpolator.set_name("state_interpolator")
        self.builder.Connect(
            adder_integrator_ik.get_output_port(),
            self.state_interpolator.get_input_port()
        )
        self.builder.ExportOutput(
            self.state_interpolator.get_output_port(), "panda_desired_state"
        )

        # Feed to controller
        self.builder.Connect(
            self.state_interpolator.get_output_port(),
            self.panda_controller.get_input_port_desired_state()
        )

    def set_hand_controller(self):
        self.hand_controller = self.builder.AddSystem(
            SchunkWsgPositionController(
                kp_command=400,  # 400
                kd_command=40,  # 40
                default_force_limit=200
            )
        )
        self.hand_controller.set_name("hand_controller")
        demux = self.builder.AddSystem(Demultiplexer(2, 1))
        self.builder.Connect(
            self.hand_controller.get_generalized_force_output_port(),
            demux.get_input_port()
        )
        self.builder.Connect(
            demux.get_output_port(0),
            # self.hand_controller.get_generalized_force_output_port(),
            self.plant.get_actuation_input_port(self.hand)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.hand),
            self.hand_controller.get_state_input_port()
        )
        hand_mbp_state_to_hand_state = self.builder.AddSystem(
            MakeMultibodyStateToWsgStateSystem()
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.hand),
            hand_mbp_state_to_hand_state.get_input_port()
        )
        self.builder.ExportOutput(
            hand_mbp_state_to_hand_state.get_output_port(),
            "hand_state_measured"
        )
        self.builder.ExportOutput(
            self.hand_controller.get_grip_force_output_port(),
            "hand_force_measured"
        )

        # add passthrough for hand position
        hand_source = self.builder.AddSystem(PassThrough(1))
        hand_source.set_name("hand_position_command_pt")
        self.builder.Connect(
            hand_source.get_output_port(),
            self.hand_controller.GetInputPort("desired_position")
        )
        self.builder.ExportInput(
            hand_source.get_input_port(), "hand_position_command"
        )

    ############################## Helper ##############################

    def AddModelFromFile(self, path, name=None):
        parser = Parser(self.plant)
        if name is None:
            num = str(len(self.object_ids))
            name = "added_model_" + num
        model_index = parser.AddModelFromFile(path, name)
        indices = self.plant.GetBodyIndices(model_index)
        return model_index, indices

    ############################## Setter ##############################

    def set_panda_position(self, station_context, q):
        num_panda_positions = self.plant.num_positions(self.panda)
        assert len(
            q
        ) == num_panda_positions, "Incorrect size of q, needs to be 7"
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        self.plant.SetPositions(plant_context, self.panda, q)

    def set_panda_velocity(self, station_context, state, v):
        num_panda_positions = self.plant.num_positions(self.panda)
        assert len(
            v
        ) == num_panda_positions, "Incorrect size of v, needs to be 7"
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetVelocities(plant_context, plant_state, self.panda, v)

    def set_hand_position(self, station_context, q):
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        self.plant.SetPositions(plant_context, self.hand, [q / 2.0, q / 2.0])

    def set_hand_velocity(self, station_context, state, v):
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        plant_state = self.GetMutableSubsystemState(self.plant, state)
        self.plant.SetVelocities(
            plant_context, plant_state, self.hand, [v / 2.0, v / 2.0]
        )

    ############################## Getter ##############################

    def get_panda_position(self, station_context):
        plant_context = self.GetSubsystemContext(self.plant, station_context)
        return self.plant.GetPositions(plant_context, self.panda)

    def get_sg(self):
        return self.sg

    def get_multibody_plant(self):
        return self.plant

    def get_controller_plant(self):
        return self.controller_plant

    def get_renderer(self):
        return self.renderer

    def get_arm_controller(self):
        return self.panda_controller

    def get_hand_controller(self):
        if hasattr(self, 'hand_controller'):
            return self.hand_controller
        else:
            return None

    def get_camera_info(self):
        return self.camera_info

    def get_panda(self):
        return self.panda

    def get_panda_c(self):
        return self.panda_c

    def get_hand(self):
        return self.hand

    def get_state_integrator(self):
        return self.state_integrator

    def get_state_interpolator(self):
        return self.state_interpolator
