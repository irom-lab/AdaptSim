from omegaconf import OmegaConf

from adaptsim.util.numeric import sample_uniform


class Joint():

    def __init__(
        self,
        rng,
        X,
        Y,
        Z,
        ROLL,
        PITCH,
        YAW,
    ):
        self.rng = rng
        self.X = X
        self.Y = Y
        self.Z = Z
        self.ROLL = ROLL
        self.PITCH = PITCH
        self.YAW = YAW

    def sample(self):
        cfg = OmegaConf.create()
        cfg.x = sample_uniform(self.rng, self.X)
        cfg.y = sample_uniform(self.rng, self.Y)
        cfg.z = sample_uniform(self.rng, self.Z)
        cfg.roll = sample_uniform(self.rng, self.ROLL)
        cfg.pitch = sample_uniform(self.rng, self.PITCH)
        cfg.yaw = sample_uniform(self.rng, self.YAW)
        return cfg
