import numpy

class LinearElastic:
    def __init__(self, E, nu, rho=0.0):
        if E <= 0:
            raise ValueError("Expected E > 0")
        if not (-1 < nu < 0.5):
            raise ValueError("Expected -1 < nu < 0.5")
        if rho < 0:
            raise ValueError("Expected rho >= 0")

        self.E = E
        self.nu = nu
        self.rho = rho

    def D_plane_stress(self):   # Matriz constitutiva D (3x3) para tension plana
        E, nu = self.E, self.nu
        c = E/(1-nu**2)
        return c * numpy.array([
            [ 1,  nu,        0],
            [nu,   1,        0],
            [ 0,   0, (1-nu)/2],])

    def D_plane_strain(self):   # Matriz constitutiva D (3x3) para deformacion plana
        E, nu = self.E, self.nu
        c = E/((1+nu)*(1-2*nu))
        return c * numpy.array([
            [1-nu,    nu,          0],
            [  nu,  1-nu,          0],
            [   0,     0, (1-2*nu)/2],])
