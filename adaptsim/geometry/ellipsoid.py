from .link import Link


class Ellipsoid(Link):

    def __init__(self, rng, DENSITY, X_DIM, Y_DIM, Z_DIM):
        super().__init__(rng, DENSITY, X_DIM, Y_DIM, Z_DIM)

    @staticmethod
    def get_volumn(x, y, z):
        return 4.1888 * x * y * z
