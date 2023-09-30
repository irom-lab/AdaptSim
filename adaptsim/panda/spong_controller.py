import numpy as np

from pydrake.all import LeafSystem, BasicVector, LinearQuadraticRegulator, Linearize
from pydrake.examples.acrobot import AcrobotPlant

from adaptsim.util.numeric import wrap_angle


class SpongController(LeafSystem):

    def __init__(self, params=None):
        LeafSystem.__init__(self)
        self.max_u = params.max_u
        k_q = params.gains['k_q']
        self.k_e = params.gains['k_e']
        self.k_p = params.gains['k_p']
        self.k_d = params.gains['k_d']
        self.balancing_threshold = 1e3

        self.acrobot_state_port = self.DeclareVectorInputPort(
            "acrobot_state", BasicVector(4)
        )
        self.DeclareVectorOutputPort(
            "tau", BasicVector(1), self.CalcOutput,
            {self.acrobot_state_port.ticket()}
        )

        # Make a new acrobot plant
        self._acrobot_plant = AcrobotPlant()
        acrobot_context = self._acrobot_plant.CreateDefaultContext()
        # Set states
        acrobot_context.SetContinuousState([np.pi, 0, 0, 0])
        # Set parameters
        acrobot_params = self._acrobot_plant.get_mutable_parameters(
            acrobot_context
        )
        # default: [1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.083, 0.33, 0.1, 0.1, 9.81]
        acrobot_params.set_m1(params.m[0])
        acrobot_params.set_m2(params.m[1])
        # acrobot_params.set_l1(1.)
        # acrobot_params.set_l2(2.)
        # acrobot_params.set_lc1(0.5)
        # acrobot_params.set_lc2(1)
        acrobot_params.set_Ic1(params.I[0])
        acrobot_params.set_Ic2(params.I[1])
        acrobot_params.set_b1(params.b[0])
        acrobot_params.set_b2(params.b[1])
        # acrobot_params.set_gravity(9.81)
        self._acrobot_context = acrobot_context

        # LQR
        self._acrobot_plant.GetInputPort('elbow_torque'
                                        ).FixValue(acrobot_context, 0.0)
        linear_system = Linearize(self._acrobot_plant, acrobot_context)
        Q = np.diag([k_q, k_q, 1, 1])
        R = [1]
        self.K, self.S = LinearQuadraticRegulator(
            linear_system.A(), linear_system.B(), Q, R
        )

    def CalcOutput(self, context, output):

        # Get current state
        x = self.acrobot_state_port.Eval(context)  # q0, q1, q0dot, q1dot
        q0, q1, q0dot, q1dot = x

        # Wrap angle
        q0 = wrap_angle(q0, 0, 2 * np.pi)
        q1 = wrap_angle(q1, -np.pi, np.pi)
        # print('after: ', q0, q1, q0dot, q1dot)

        # Get cost
        x0 = np.array([np.pi, 0, 0, 0])
        cost = np.dot(x - x0, self.S.dot(x - x0))

        # LQR if close to top
        if cost < self.balancing_threshold:
            u = self.K.dot(x0 - x)[0]

        # Pump energy!
        else:
            # update state
            self._acrobot_context.SetContinuousState(x)
            M = self._acrobot_plant.MassMatrix(self._acrobot_context)
            bias = self._acrobot_plant.DynamicsBiasTerm(self._acrobot_context)
            M_inverse = np.linalg.inv(M)

            PE = self._acrobot_plant.EvalPotentialEnergy(self._acrobot_context)
            KE = self._acrobot_plant.EvalKineticEnergy(self._acrobot_context)
            E = PE + KE

            p = self._acrobot_plant.get_parameters(self._acrobot_context)
            E_desired = (p.m1() * p.lc1() + p.m2() *
                         (p.l1() + p.lc2())) * p.gravity()
            E_tilde = E - E_desired
            u_e = -self.k_e * E_tilde * q1dot

            y = -self.k_p * q1 - self.k_d * q1dot
            a3 = M_inverse[1, 1]
            a2 = M_inverse[0, 1]
            u_p = (a2 * bias[0] + y) / a3 + bias[1]

            u = u_e + u_p

        # Saturation
        u = max(-self.max_u, u)
        u = min(self.max_u, u)

        output.SetFromVector([u])
