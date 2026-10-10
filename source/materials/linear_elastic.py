import numpy


class LinearElastic:
    def __init__(self, E, nu, rho=0.0):
        if E <= 0.0:
            raise ValueError("Expected E > 0")
        if not (-1.0 < nu < 0.5):
            raise ValueError("Expected -1 < nu < 0.5")
        if rho < 0.0:
            raise ValueError("Expected rho>=0")

        self.E = E
        self.nu = nu
        self.rho = rho

    def D_plane_stress(self):
        E, nu = self.E, self.nu
        c = E / (1.0 - nu**2)
        return c * numpy.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])

    def D_plane_strain(self):
        E, nu = self.E, self.nu
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * numpy.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ])
