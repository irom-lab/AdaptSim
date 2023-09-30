import numpy as np
import time
import logging
from pydrake.all import DiagramBuilder, Simulator, AddMultibodyPlantSceneGraph, Parser, DrakeVisualizer, LinearQuadraticRegulator, Saturation, PassThrough, Adder, Linearize


class DoublePendulumEnv():

    def __init__(
        self,
        render=False,
        dt=0.005,
        torque_limit=2e2,
        max_t=2,
        **kwargs,
    ):
        self.render = render
        self.dt = dt
        self.torque_limit = torque_limit
        self.max_step = int(max_t / dt) + 1
        self.flag_setup = False

    def _setup(self, task):
        """
        Set up double pendulum.
        """
        # Load objects
        builder = DiagramBuilder()
        self._plant, self._sg = AddMultibodyPlantSceneGraph(
            builder, time_step=self.dt
        )
        parser = Parser(self._plant, self._sg)
        model_index = parser.AddModelFromFile(
            "asset/double_pendulum/double_pendulum.sdf"
        )

        # Set fake damping for LQR
        self._joint_indices = self._plant.GetJointIndices(model_index)
        for ind, joint_index in enumerate(self._joint_indices):
            damping = task.lqr_b[ind]
            damping = max(0, damping)
            joint = self._plant.get_joint(joint_index)
            joint.set_default_damping(damping)

        # Finalize
        self._plant.Finalize()

        # Get ports and info
        self._actuation_input_port = self._plant.GetInputPort(
            "double_pendulum_actuation"
        )
        self._dummy_input_port = self._plant.GetInputPort("actuation")
        self._body_indices = self._plant.GetBodyIndices(model_index)
        self._joint_indices = self._plant.GetJointIndices(model_index)
        self.visualizer = None
        if self.render:
            self.visualizer = DrakeVisualizer.AddToBuilder(
                builder=builder,
                scene_graph=self._sg,
            )

        # Set LQR - pre-finalize - set fake mass
        linearize_context = self._plant.CreateDefaultContext()
        for ind, body_index in enumerate(self._body_indices):
            mass = task.lqr_m[ind]
            body = self._plant.get_body(body_index)
            body.SetMass(linearize_context, mass)
        linearize_context.SetDiscreteState(
            np.array([np.pi, 0., 0., 0])
        )  # linearize around upright and zero velocities
        Q = np.identity(4)
        R = np.identity(2)
        self._actuation_input_port.FixValue(linearize_context, [0, 0])

        lqr_system = LinearQuadraticRegulator(
            system=self._plant,
            context=linearize_context,
            Q=Q,
            R=R,
            N=np.zeros(0),
            input_port_index=self._actuation_input_port.get_index(),
        )
        self._lqr = builder.AddSystem(lqr_system)
        system = Linearize(
            self._plant, linearize_context,
            input_port_index=self._actuation_input_port.get_index(),
            output_port_index=self._plant.get_state_output_port().get_index()
        )
        sym_plant = self._plant.ToSymbolic()

        # Apply the torque limit.
        torque_limiter = builder.AddSystem(
            Saturation(
                min_value=np.ones((2)) * -self.torque_limit,
                max_value=np.ones((2)) * self.torque_limit
            )
        )

        # Add torque offset to realize true damping
        torque_offset = builder.AddSystem(PassThrough([0] * 2))
        builder.ExportInput(torque_offset.get_input_port(), "torque_offset")
        adder = builder.AddSystem(Adder(2, 2))
        builder.Connect(
            torque_offset.get_output_port(), adder.get_input_port(0)
        )
        builder.Connect(
            torque_limiter.get_output_port(), adder.get_input_port(1)
        )

        # Wire
        builder.Connect(
            self._lqr.get_output_port(0), torque_limiter.get_input_port(0)
        )
        builder.Connect(adder.get_output_port(0), self._actuation_input_port)
        builder.Connect(
            self._plant.get_state_output_port(), self._lqr.get_input_port(0)
        )
        builder.ExportOutput(adder.get_output_port(0), "actuation")

        # Set up simulator
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.Publish(context)
        self.simulator = Simulator(diagram, context)

        # Get torque offset port from diagram
        self.torque_offset_port = diagram.GetInputPort('torque_offset')
        self.actuation_port = diagram.GetOutputPort('actuation')

        # Get diagram PDF
        # import pydotplus
        # pydot_graph = pydotplus.graph_from_dot_data(diagram.GetGraphvizString())
        # pydot_graph.write_pdf("double_pendulum_diagram.pdf")

    def reset(self, task=None):
        """
        Set up simulator if first time. Reset task.
        """
        # Save task
        if task is not None:
            self.task = task

        # Reset pendulum and LQR if first time or task includes LQR parameters
        if not self.flag_setup or 'lqr_m' in task:
            self._setup(task)  # 0.07s to run
            self.flag_setup = True

        # Set task - true mass and true damping (by torque offset)
        self.reset_task(task)

        # Reset simulation - reset pendulum states
        context = self.simulator.get_mutable_context()
        context.SetTime(0.)
        plant_context = self._plant.GetMyContextFromRoot(context)
        self.set_states(plant_context, [0, 0])
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
        for ind, body_index in enumerate(self._body_indices):
            body = self._plant.get_body(body_index)
            body.SetMass(plant_context, task.true_m[ind])

        # Record true damping - find offset
        self.b_offset = np.array(task.true_b) - np.array(task.lqr_b)

    @property
    def parameter(self):
        return self.task.true_m + self.task.true_b

    def step(self, action):
        """
        Run the entire episode with LQR. No input action. Use info to return trajectory information.
        """
        info = {}
        q_snapshot = []
        a_snapshot = []
        total_a = 0
        for step_ind in range(1, self.max_step):

            context = self.simulator.get_mutable_context()
            plant_context = self._plant.GetMyContextFromRoot(context)

            # Get state and action
            q, qdot = self.get_states(plant_context)
            a = self.actuation_port.Eval(context)

            # Set input - add damping manually
            torque_offset = -self.b_offset * qdot
            self.torque_offset_port.FixValue(context, torque_offset)

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
            total_a += (abs(a[0]) + abs(a[1])) / self.max_step

            # Save q
            q_snapshot += [q]
            a_snapshot += [a]

            if self.render:
                time.sleep(0.01)

        # Add q to info
        info['q'] = q_snapshot
        info['a'] = a_snapshot
        info['param'] = self.parameter

        # Get reward
        target_ee_z = 2
        ee_z = self.get_ee_z(plant_context)
        ee_z_diff = np.abs(target_ee_z - ee_z)
        reward = 4 - ee_z_diff
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
        return q, qdot

    def set_states(self, context, q, qdot=[0] * 2):
        self._plant.SetPositions(context, q)
        self._plant.SetVelocities(context, qdot)

    def get_ee_z(self, plant_context):
        """Assume each link has length 1
        """
        q = self.get_states(plant_context)[0]
        p1 = -1 * np.cos(q[0])
        q2 = q[0] + q[1]
        p2 = p1 - 1 * np.cos(q2)
        return p2