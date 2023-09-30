"""
Following libfranka: https://github.com/frankaemika/libfranka/blob/master/src/lowpass_filter.cpp

"""
from pydrake.all import LeafSystem, BasicVector
import numpy as np


class DiscreteLowPassFilter(LeafSystem):

    def __init__(self, dt, cutoff=100):
        LeafSystem.__init__(self)
        self._dt = dt
        self._cutoff_freq = cutoff

        self.qdot_d_port = self.DeclareVectorInputPort(
            "qdot_d", BasicVector(7)
        )
        self.DeclareVectorOutputPort(
            "qdot_filtered", BasicVector(7), self.CalcOutput,
            {self.qdot_d_port.ticket()}
        )
        self.last_command = np.zeros((7))

    def CalcOutput(self, context, output):
        qdot_d = self.qdot_d_port.Eval(context)

        gain = self._dt / (
            self._dt + (1.0 / (2.0 * np.pi * self._cutoff_freq))
        )
        qdot_filtered = gain*qdot_d + (1-gain) * self.last_command
        self.last_command = qdot_filtered

        output.SetFromVector(qdot_filtered)
