import numpy

from source.elements.quadrature import triangle_rule, quad_rule


class ReferenceElement:
    name = ""
    meshio_type = ""        # nombre de celda para meshio / ParaView
    node_coords = []
    is_triangle = False

    def number_of_nodes(self):
        return len(self.node_coords)

    def reference_area(self):
        return 0.5 if self.is_triangle else 4.0

    def shape(self, xi, eta):
        """Devuelve N (nn,) y dN (nn,2) = [dN/dxi, dN/deta]."""
        raise NotImplementedError

    def gauss_rule(self):
        """Devuelve puntos (ng,2) y pesos (ng,)."""
        raise NotImplementedError


class Tri3(ReferenceElement):
    name = "TRI3"
    meshio_type = "triangle"
    node_coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    is_triangle = True

    def shape(self, xi, eta):
        L = 1.0 - xi - eta
        N = numpy.array([L, xi, eta])
        dN = numpy.array([[-1.0, -1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        return N, dN

    def gauss_rule(self):
        return triangle_rule(1)


class Tri6(ReferenceElement):
    name = "TRI6"
    meshio_type = "triangle6"
    node_coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                   (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    is_triangle = True

    def shape(self, xi, eta):
        L = 1.0 - xi - eta
        N = numpy.array([
            L * (2 * L - 1),
            xi * (2 * xi - 1),
            eta * (2 * eta - 1),
            4 * L * xi,
            4 * xi * eta,
            4 * eta * L,
        ])
        dN = numpy.array([
            [-(4 * L - 1), -(4 * L - 1)],
            [4 * xi - 1, 0.0],
            [0.0, 4 * eta - 1],
            [4 * (L - xi), -4 * xi],
            [4 * eta, 4 * xi],
            [-4 * eta, 4 * (L - eta)],
        ])
        return N, dN

    def gauss_rule(self):
        return triangle_rule(3)


class Quad4(ReferenceElement):
    name = "QUAD4"
    meshio_type = "quad"
    node_coords = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    def shape(self, xi, eta):
        N = numpy.zeros(4)
        dN = numpy.zeros((4, 2))

        for i, (xi_i, eta_i) in enumerate(self.node_coords):
            N[i] = 0.25 * (1 + xi_i * xi) * (1 + eta_i * eta)
            dN[i, 0] = 0.25 * xi_i * (1 + eta_i * eta)
            dN[i, 1] = 0.25 * eta_i * (1 + xi_i * xi)

        return N, dN

    def gauss_rule(self):
        return quad_rule(2)


class Quad8(ReferenceElement):
    name = "QUAD8"
    meshio_type = "quad8"
    node_coords = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
                   (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]

    def shape(self, xi, eta):
        N = numpy.zeros(8)
        dN = numpy.zeros((8, 2))

        for i, (xi_i, eta_i) in enumerate(self.node_coords):
            if i < 4:                       # esquinas
                a = 1 + xi_i * xi
                b = 1 + eta_i * eta
                c = xi_i * xi + eta_i * eta - 1
                N[i] = 0.25 * a * b * c
                dN[i, 0] = 0.25 * xi_i * b * (c + a)
                dN[i, 1] = 0.25 * eta_i * a * (c + b)
            elif xi_i == 0.0:               # medio de lado inferior/superior
                b = 1 + eta_i * eta
                N[i] = 0.5 * (1 - xi**2) * b
                dN[i, 0] = -xi * b
                dN[i, 1] = 0.5 * (1 - xi**2) * eta_i
            else:                           # medio de lado izquierdo/derecho
                a = 1 + xi_i * xi
                N[i] = 0.5 * a * (1 - eta**2)
                dN[i, 0] = 0.5 * xi_i * (1 - eta**2)
                dN[i, 1] = -eta * a

        return N, dN

    def gauss_rule(self):
        return quad_rule(3)


REFERENCE_ELEMENTS = {
    "TRI3": Tri3,
    "TRI6": Tri6,
    "QUAD4": Quad4,
    "QUAD8": Quad8,
}


def reference_element(name):
    if name not in REFERENCE_ELEMENTS:
        raise ValueError("Unknown element type: " + name)

    return REFERENCE_ELEMENTS[name]()

