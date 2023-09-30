import numpy as np
import logging
from omegaconf import OmegaConf
from pydrake.all import DiagramBuilder, Simulator, Parser, DrakeVisualizer, MultibodyPlantConfig, AddMultibodyPlant

from adaptsim.panda.spong_controller import SpongController
from adaptsim.util.numeric import wrap_angle


class AcrobotEnv():

    def __init__(
        self,
        render=False,
        dt=0.01,
        max_t=2,
    ):
        self.render = render
        self.max_step = int(max_t / dt) + 1
        self.dt = dt

    def _setup(self, task):
        """
        Set up acrobot.

        """
        # Load objects
        builder = DiagramBuilder()
        plant_cfg = MultibodyPlantConfig(
            time_step=self.dt, discrete_contact_solver='sap'
        )
        self._plant, self._sg = AddMultibodyPlant(plant_cfg, builder)
        parser = Parser(self._plant, self._sg)
        model_index = parser.AddModelFromFile("asset/acrobot/acrobot.sdf")

        # Set damping - both joints
        self._joint_indices = self._plant.GetJointIndices(model_index)
        for ind, joint_index in enumerate(self._joint_indices[1:]):
            damping = max(0, task.true_b[ind])
            joint = self._plant.get_joint(joint_index)
            joint.set_default_damping(damping)

        # Finalize
        self._plant.Finalize()

        # Get ports and info
        self._actuation_input_port = self._plant.GetInputPort(
            "acrobot_actuation"
        )
        self._body_indices = self._plant.GetBodyIndices(model_index)
        self._joint_indices = self._plant.GetJointIndices(model_index)
        self.visualizer = None
        if self.render:
            self.visualizer = DrakeVisualizer.AddToBuilder(
                builder=builder,
                scene_graph=self._sg,
            )

        # Add controller plant using Drake's Acrobot implementation
        cfg = OmegaConf.create()
        cfg.m = task.spong_m
        cfg.b = task.spong_b
        cfg.I = [
            0.083 * task.spong_m[0], 0.33 * task.spong_m[1]
        ]  # with m=1 for both links, I=0.083 and 0.33, and scale linearly
        cfg.max_u = 40  # default was 20 for m=1
        cfg.gains = task.gains
        self._controller = builder.AddSystem(SpongController(cfg))

        # Wire
        builder.Connect(
            self._plant.get_state_output_port(),
            self._controller.GetInputPort('acrobot_state')
        )
        builder.Connect(
            self._controller.get_output_port(0), self._actuation_input_port
        )

        # Wire
        builder.ExportOutput(self._controller.get_output_port(0), "actuation")

        # Set up simulator
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.Publish(context)
        self.simulator = Simulator(diagram, context)

        # Get torque offset port from diagram
        self.actuation_port = diagram.GetOutputPort('actuation')

        # Get diagram PDF
        # import pydotplus
        # pydot_graph = pydotplus.graph_from_dot_data(diagram.GetGraphvizString())
        # pydot_graph.write_pdf("acrobot_new_diagram.pdf")

    def reset(self, task=None):
        """
        Set up simulator if first time. Reset task.
        """
        # Save task
        if task is not None:
            self.task = task

        # Reset, including controller
        self._setup(task)

        # Set task - true mass
        self.reset_task(task)

        # Reset simulation - reset acrobot states
        context = self.simulator.get_mutable_context()
        context.SetTime(0.)
        plant_context = self._plant.GetMyContextFromRoot(context)
        self.set_states(plant_context, [1, 0])

        # Initialize simulator
        self.simulator.Initialize()
        self._t = 0
        return self._get_obs()

    def reset_task(self, task=None):
        """
        Reset the task for the environment.
        """
        context = self.simulator.get_mutable_context()
        plant_context = self._plant.GetMyContextFromRoot(context)

        # Set true mass post-finalize - also sets the spatial inertia
        for ind, body_index in enumerate(self._body_indices[1:]):
            body = self._plant.get_body(body_index)
            body.SetMass(plant_context, task.true_m[ind])

    @property
    def parameter(self):
        return self.task.true_m + self.task.true_b

    def step(self, action):
        """
        Run the entire episode with LQR. No input action. Use info to return trajectory information.
        """
        height_threshold = 2.5

        info = {}
        x_snapshot = []
        a_snapshot = []
        total_a = 0
        num_step_above_threshold = 0
        for step_ind in range(1, self.max_step):

            context = self.simulator.get_mutable_context()
            plant_context = self._plant.GetMyContextFromRoot(context)

            # Get state and action
            x = self.get_states(plant_context)
            a = self.actuation_port.Eval(context)

            # Simulate - assume success and not catching error
            self._t = step_ind * self.dt
            try:
                status = self.simulator.AdvanceTo(self._t)
            except:
                logging.error(
                    f'Sim error at time {self._t} with parameters {self.parameter}!'
                )
                raise

            # Accumulate effort
            total_a += abs(a[0]) / self.max_step

            # Save state and action - angle wrapped
            x[0] = wrap_angle(x[0], -np.pi, np.pi)
            x[1] = wrap_angle(x[1], -np.pi, np.pi)
            x_snapshot += [x]
            a_snapshot += [a]

            # Check if above threshold
            ee_z = self.get_ee_z(plant_context)
            if ee_z > height_threshold:
                num_step_above_threshold += 1

            if self.render:
                import time
                time.sleep(0.01)

        # Add state/action/parameter
        info['x'] = x_snapshot
        info['a'] = a_snapshot
        info['param'] = self.parameter

        # Get reward
        reward = num_step_above_threshold / self.max_step
        done = True
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.ones((2, 2))

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        return [seed]

    def get_states(self, context):
        q = self._plant.GetPositions(context)
        qdot = self._plant.GetVelocities(context)
        return np.hstack((q, qdot))

    def set_states(self, context, q, qdot=[0] * 2):
        self._plant.SetPositions(context, q)
        self._plant.SetVelocities(context, qdot)

    def get_ee_z(self, plant_context):
        """Assume each link has length 1
        """
        q = self.get_states(plant_context)[0:2]
        p1 = -1 * np.cos(q[0])
        q2 = q[0] + q[1]
        p2 = p1 - 2 * np.cos(q2)
        return p2