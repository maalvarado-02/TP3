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

    def shape(self, xi, eta):  # Devuelve N (nn,) y dN (nn,2) = [dN/dxi, dN/deta].
        raise NotImplementedError

    def gauss_rule(self):   # Devuelve puntos (ng,2) y pesos (ng,).
        raise NotImplementedError


class Tri3(ReferenceElement):
    name = "TRI3"
    meshio_type = "triangle"
    node_coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    is_triangle = True

    def shape(self, xi, eta):
        N1 = 1.0 - xi - eta
        N2 = xi
        N3 = eta
        N = numpy.array([N1, N2, N3])
        dN1 = [-1.0, -1.0]
        dN2 = [1.0, 0.0]
        dN3 = [0.0, 1.0]
        dN = numpy.array([dN1, dN2, dN3])
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
        N1 = L * (2 * L - 1)
        N2 = xi * (2 * xi - 1)
        N3 = eta * (2 * eta - 1)
        N4 = 4 * L * xi
        N5 = 4 * xi * eta
        N6 = 4 * eta * L
        N = numpy.array([N1, N2, N3, N4, N5, N6])
        dN1 = [-4 * L + 1, -4 * L + 1]
        dN2 = [4 * xi - 1, 0.0]
        dN3 = [0.0, 4 * eta - 1]
        dN4 = [4 * (L - xi), -4 * xi]
        dN5 = [4 * eta, 4 * xi]
        dN6 = [-4 * eta, 4 * (L - eta)]
        dN = numpy.array([dN1, dN2, dN3, dN4, dN5, dN6])
        return N, dN

    def gauss_rule(self):
        return triangle_rule(3)


class Quad4(ReferenceElement):
    name = "QUAD4"
    meshio_type = "quad"
    node_coords = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    def shape(self, xi, eta):
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        N = numpy.array([N1, N2, N3, N4])
        dN1 = [-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)]
        dN2 = [0.25 * (1.0 - eta), -0.25 * (1.0 + xi)]
        dN3 = [0.25 * (1.0 + eta), 0.25 * (1.0 + xi)]
        dN4 = [-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)]
        dN = numpy.array([dN1, dN2, dN3, dN4])
        return N, dN

    def gauss_rule(self):
        return quad_rule(2)


class Quad8(ReferenceElement):
    name = "QUAD8"
    meshio_type = "quad8"
    node_coords = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
                   (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]

    def shape(self, xi, eta):
        N1 = -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta)
        N2 = -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta)
        N3 = -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta)
        N4 = -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta)
        N5 = 0.5 * (1 - xi**2) * (1 - eta)
        N6 = 0.5 * (1 + xi) * (1 - eta**2)
        N7 = 0.5 * (1 - xi**2) * (1 + eta)
        N8 = 0.5 * (1 - xi) * (1 - eta**2)
        N = numpy.array([N1, N2, N3, N4, N5, N6, N7, N8])
        dN1 = [0.25 * (1 - eta) * (2 * xi + eta),  0.25 * (1 - xi) * (xi + 2 * eta)]
        dN2 = [0.25 * (1 - eta) * (2 * xi - eta), -0.25 * (1 + xi) * (xi - 2 * eta)]
        dN3 = [0.25 * (1 + eta) * (2 * xi + eta),  0.25 * (1 + xi) * (xi + 2 * eta)]
        dN4 = [0.25 * (1 + eta) * (2 * xi - eta), -0.25 * (1 - xi) * (xi - 2 * eta)]
        dN5 = [-xi * (1 - eta), -0.5 * (1 - xi**2)]
        dN6 = [0.5 * (1 - eta**2), -(1 + xi) * eta]
        dN7 = [-xi * (1 + eta), 0.5 * (1 - xi**2)]
        dN8 = [-0.5 * (1 - eta**2), -(1 - xi) * eta]
        dN = numpy.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])
        return N, dN

    def gauss_rule(self):
        return quad_rule(3)


REFERENCE_ELEMENTS = {
    "TRI3": Tri3,
    "TRI6": Tri6,
    "QUAD4": Quad4,
    "QUAD8": Quad8,}


def reference_element(name):
    if name not in REFERENCE_ELEMENTS:
        raise ValueError("Unknown element type: " + name)

    return REFERENCE_ELEMENTS[name]()