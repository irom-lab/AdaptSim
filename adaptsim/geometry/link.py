from omegaconf import OmegaConf

from adaptsim.util.numeric import sample_uniform


class Link():
    """Base
    """

    def __init__(
        self,
        rng,
        DENSITY,
        X_DIM,  # all half dimensions
        Y_DIM,
        Z_DIM,
        #  color,
    ):
        self.rng = rng
        self.DENSITY = DENSITY
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        # self.color = color

    def sample(self,):
        cfg = OmegaConf.create()
        cfg.x = sample_uniform(self.rng, self.X_DIM)
        cfg.y = sample_uniform(self.rng, self.Y_DIM)
        cfg.z = sample_uniform(self.rng, self.Z_DIM)
        density = sample_uniform(self.rng, self.DENSITY)
        cfg.m = density * 1e3 * self.get_volumn(cfg.x, cfg.y, cfg.z)
        cfg.density = density
        return cfg

    # def sample_color(self):
    #     """Sample the color.

    #     Returns:
    #         Color as values of (r, g, b).
    #     """
    #     if self.color is None:
    #         color = np.random.rand(3)
    #     else:
    #         color = self.color

    #     return color
