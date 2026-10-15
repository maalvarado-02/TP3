import numpy

from source.elements.quadrature import triangle_rule, quad_rule, line_rule

# Armado de funciones de forma N y derivadas dN para elementos 2D en coord parametricas (xi, eta) y aristas de 2 y 3 nodos

class ReferenceElement:   # Clase base para elementos de referencia 2D (triangulos y cuadrilateros)
    name = ""
    meshio_type = ""   # Nombre de celda para meshio/ParaView
    node_coords = []   # Coord de los nodos
    edges = []         # Indices locales de nodo (aristas antihorarias)
    is_triangle = False

    def number_of_nodes(self):   # Devuelve numero de nodos del elemento
        return len(self.node_coords)

    def reference_area(self):   # Devuelve area del elemento
        return 0.5 if self.is_triangle else 4.0

    def shape(self, xi, eta):  # Devuelve N(nn) y dN(nn,2) = [dN/dxi, dN/deta].
        raise NotImplementedError

    def gauss_rule(self):   # Devuelve puntos(ng,2) y pesos(ng).
        raise NotImplementedError

    def edge_element(self):   # Devuelve 
        return line_element(len(self.edges[0]))


class LineElement:   # Clase base para elementos de referencia 1D en s en [-1, 1]
    name = ""
    node_coords = []

    def number_of_nodes(self):
        return len(self.node_coords)

    def shape(self, s):        # N (nn,) y dN (nn,) = dN/ds
        raise NotImplementedError

    def gauss_rule(self):      # puntos (ng,) y pesos (ng,)
        raise NotImplementedError


# Elemento triangular lineal de 3 nodos
#-------------------------------------------------
class Tri3(ReferenceElement):
    name = "TRI3"
    edges = [(0, 1), (1, 2), (2, 0)]
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

    def gauss_rule(self):   # Regla de Gauss de 1 punto
        return triangle_rule(1)


# Elemento triangular cuadratico de 6 nodos
#-------------------------------------------------
class Tri6(ReferenceElement):
    name = "TRI6"
    edges = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
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

    def gauss_rule(self):   # Regla de Gauss de 3 puntos
        return triangle_rule(3)


# Elemento cuadrilatero bilineal de 4 nodos
#-------------------------------------------------
class Quad4(ReferenceElement):
    name = "QUAD4"
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
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

    def gauss_rule(self):   # Regla de Gauss de 2x2 puntos
        return quad_rule(2)


# Elemento cuadrilatero cuadratico de 8 nodos
#-------------------------------------------------
class Quad8(ReferenceElement):
    name = "QUAD8"
    edges = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
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

    def gauss_rule(self):   # Regla de Gauss de 3x3 puntos
        return quad_rule(3)


# 
#-------------------------------------------------
class Line2(LineElement):      # orden: [a, b]
    name = "LINE2"
    node_coords = [-1.0, 1.0]

    def shape(self, s):
        Na = 0.5 * (1 - s)
        Nb = 0.5 * (1 + s)
        N = numpy.array([Na, Nb])
        dNa = -0.5
        dNb = 0.5
        dN = numpy.array([dNa, dNb])
        return N, dN

    def gauss_rule(self):
        return line_rule(2)


# 
#-------------------------------------------------
class Line3(LineElement):      # orden: [a, b, lado]
    name = "LINE3"
    node_coords = [-1.0, 1.0, 0.0]

    def shape(self, s):
        Na = 0.5 * s * (s - 1)
        Nb = 0.5 * s * (s + 1)
        Nlado = 1 - s * s
        N = numpy.array([Na, Nb, Nlado])
        dNa = s - 0.5
        dNb = s + 0.5
        dNlado = -2 * s
        dN = numpy.array([dNa, dNb, dNlado])
        return N, dN

    def gauss_rule(self):
        return line_rule(3)


REFERENCE_ELEMENTS = {"TRI3": Tri3, "TRI6": Tri6, "QUAD4": Quad4, "QUAD8": Quad8,}
LINE_ELEMENTS = {2: Line2, 3: Line3}

def reference_element(name):
    if name not in REFERENCE_ELEMENTS:
        raise ValueError("Unknown element type: " + name)

    return REFERENCE_ELEMENTS[name]()

def line_element(number_of_nodes):
    if number_of_nodes not in LINE_ELEMENTS:
        raise ValueError("Unsupported edge with %d nodes" % number_of_nodes)
    return LINE_ELEMENTS[number_of_nodes]()