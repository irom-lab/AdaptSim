from pydrake.all import LeafSystem, BasicVector
import numpy as np


class RateLimiter(LeafSystem):

    def __init__(self, dt, flag_disable=False):
        LeafSystem.__init__(self)
        self._dt = dt
        self.flag_disable = flag_disable

        self.qdot_d_port = self.DeclareVectorInputPort(
            "qdot_d", BasicVector(7)
        )
        self.qdot_port = self.DeclareVectorInputPort("qdot", BasicVector(7))
        self.DeclareVectorOutputPort(
            "qdot_limited", BasicVector(7), self.CalcOutput,
            {self.qdot_d_port.ticket()}
        )

        self.acc_limit = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
        self.vel_limit = np.array([
            2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610
        ])

    def CalcOutput(self, context, output):
        qdot_d = self.qdot_d_port.Eval(context)
        qdot = self.qdot_port.Eval(context)
        # q = self.q_port.Eval(context)

        if not self.flag_disable:
            # Clip acceleration
            # a_lb = lambda v, b: -b*self._dt + v
            # a_ub = lambda v, b: b*self._dt + v
            upper = qdot + self.acc_limit * self._dt
            lower = qdot - self.acc_limit * self._dt
            qdot_d = np.clip(qdot_d, lower, upper)

            # Clip velocity
            qdot_d = np.clip(qdot_d, -self.vel_limit, self.vel_limit)

            # for i in range(7):
            #     qdot_d[i] = max(qdot_d[i], a_lb(qdot[i], acc_limit[i]))
            #     qdot_d[i] = min(qdot_d[i], a_ub(qdot[i], acc_limit[i]))

            # Clip velocity
            # for i in range(7):
            #     qdot_d[i] = max(qdot_d[i], -vel_limit[i])
            # qdot_d[i] = min(qdot_d[i], vel_limit[i])

        # Ignore jerk limits: [7500, 3750, 5000, 6250, 7500, 10000, 10000]
        output.SetFromVector(qdot_d)
