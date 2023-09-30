from .link import Link


class Cylinder(Link):

    def __init__(self, rng, DENSITY, R_DIM, Z_DIM):
        super().__init__(
            rng,
            DENSITY,
            R_DIM,  # use radius for both x and y
            R_DIM,
            Z_DIM
        )

    @staticmethod
    def get_volumn(x, y, z):
        return 3.1416 * (x**2) * (2*z)
