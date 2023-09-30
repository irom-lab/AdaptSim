import numpy as np

from pydrake.all import LeafSystem, BasicVector, JacobianWrtVariable


class PseudoInverseController(LeafSystem):

    def __init__(self, plant, dt):
        LeafSystem.__init__(self)
        self._dt = dt
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._panda = plant.GetModelInstanceByName("panda")
        self._G = plant.GetBodyByName("panda_link8").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", BasicVector(6))
        self.q_port = self.DeclareVectorInputPort("q", BasicVector(7))
        self.qdot_port = self.DeclareVectorInputPort("qdot", BasicVector(7))
        self.DeclareVectorOutputPort(
            "panda_velocity", BasicVector(7), self.CalcOutput,
            {self.V_G_port.ticket()}
        )
        self.panda_start = plant.GetJointByName("panda_joint1"
                                               ).velocity_start()
        self.panda_end = plant.GetJointByName("panda_joint7").velocity_start()

        # Prev solution for initial guess
        self.prev_sol = np.zeros((7))
        self.prev_v_g = np.zeros((6))
        self.prev_qdot = np.zeros((6))

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        qdot = self.qdot_port.Eval(context)

        # Return zero if V_G all zero - do not calculate
        if np.all(np.abs(V_G) < 1e-5):
            output.SetFromVector(np.zeros((7)))
            return

        # Solve if first time in a time step - otherwise just return the last result
        self.q = q
        self.qdot = qdot

        # Set states for controller plant?
        self._plant.SetPositions(self._plant_context, self._panda, self.q)
        self._plant.SetVelocities(self._plant_context, self._panda, self.qdot)

        # Update jacobian and current spatial velocities
        self.J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV, self._G, [0, 0, 0],
            self._W, self._W
        )  # ang, then trans
        self.J_G = self.J_G[:, self.panda_start:self.panda_end + 1]
        self.cur_v = self.J_G.dot(self.qdot)

        # Debug: check manipulability
        # m = np.sqrt(np.linalg.det(self.J_G.dot(self.J_G.T)))
        # print('Manipulability: ', m)

        # Option 1: psuedo-inverse
        # v = np.linalg.pinv(self.J_G).dot(V_G)
        # output.SetFromVector(v)

        # Option 2: damped psuedo-inverse
        damping = np.eye((6)) * 0.002
        pinv = self.J_G.T.dot(
            np.linalg.inv(self.J_G.dot(self.J_G.T) + damping)
        )
        v = pinv.dot(V_G)

        # Option 3: nullspace - e.g. minimize joint 1 velocity
        # pinv = np.linalg.pinv(self.J_G)
        # P = np.eye(7) - pinv.dot(self.J_G)
        # c = np.array([0, 1e3, 0, 0, 0, 0, 0])
        # off = P.dot(c)
        # v = pinv.dot(V_G) - off

        # Update
        output.SetFromVector(v)
        self.prev_sol = v
        self.prev_v_g = V_G
        self.prev_qdot = qdot

        # Option 2: QP with constraints
        # prog = MathematicalProgram()
        # alpha = prog.NewContinuousVariables(1)
        # v = prog.NewContinuousVariables(7)
        # # s = prog.NewContinuousVariables(7)
        # # residual = J_G.dot(v) - alpha[0]*V_G
        # residual = self.J_G.dot(v) - alpha[0]*V_G
        # residual = self.J_G.dot(v) - (alpha[0]*V_G + (1-alpha[0])*self.cur_v)
        # prog.AddCost(-alpha[0]) # max
        # prog.AddBoundingBoxConstraint(0, 1, alpha[0])
        # # prog.AddQuadraticErrorCost(0.1*np.identity(7), np.zeros((7)), v)
        # # prog.AddQuadraticErrorCost(np.identity(7), np.zeros((7)), s)
        # # prog.Add2NormSquaredCost(self.J_G, V_G, v)
        # # prog.AddCost(residual.dot(residual)) # max
        # prog.AddLinearConstraint(residual[0] == 0)
        # prog.AddLinearConstraint(residual[1] == 0)
        # prog.AddLinearConstraint(residual[2] == 0)
        # prog.AddLinearConstraint(residual[3] == 0)
        # prog.AddLinearConstraint(residual[4] == 0)
        # prog.AddLinearConstraint(residual[5] == 0)

        # # Joint velocity constraints - ignore joint position constraint for now
        # prog.AddBoundingBoxConstraint(-2.175, 2.175, v[0])
        # prog.AddBoundingBoxConstraint(-2.175, 2.175, v[1])
        # prog.AddBoundingBoxConstraint(-2.175, 2.175, v[2])
        # prog.AddBoundingBoxConstraint(-2.175, 2.175, v[3])
        # prog.AddBoundingBoxConstraint(-2.610, 2.610, v[4])
        # prog.AddBoundingBoxConstraint(-2.610, 2.610, v[5])
        # prog.AddBoundingBoxConstraint(-2.610, 2.610, v[6])

        # # Joint acceleration constraints
        # a_lb = lambda v, b: -b*self._dt + v
        # a_ub = lambda v, b: b*self._dt + v
        # qdot = self.qdot
        # prog.AddBoundingBoxConstraint(a_lb(qdot[0], 15),
        #                               a_ub(qdot[0], 15),
        #                               v[0])
        # prog.AddBoundingBoxConstraint(a_lb(qdot[1], 7.5),
        #                               a_ub(qdot[1], 7.5),
        #                               v[1])
        # prog.AddBoundingBoxConstraint(a_lb(qdot[2], 10),
        #                               a_ub(qdot[2], 10),
        #                               v[2])
        # prog.AddBoundingBoxConstraint(a_lb(qdot[3], 12.5),
        #                               a_ub(qdot[3], 12.5),
        #                               v[3])
        # prog.AddBoundingBoxConstraint(a_lb(qdot[4], 15),
        #                               a_ub(qdot[4], 15),
        #                               v[4])
        # prog.AddBoundingBoxConstraint(a_lb(qdot[5], 20),
        #                               a_ub(qdot[5], 20),
        #                               v[5])
        # prog.AddBoundingBoxConstraint(a_lb(qdot[6], 20),
        #                               a_ub(qdot[6], 20),
        #                               v[6])
