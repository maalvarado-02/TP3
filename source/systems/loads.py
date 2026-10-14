import numpy

from source.elements.quadrature import line_rule

# Aristas de cada tipo (indices locales, sentido antihorario):
# (esquina_a, esquina_b) (T3 y Q4) o  (esquina_a, esquina_b, nodo_de_lado) (T6 y Q8)
EDGES = {
    "TRI3": [(0, 1), (1, 2), (2, 0)],
    "TRI6": [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
    "QUAD4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "QUAD8": [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],}


# Fuerzas de cuerpo (peso propio)
# ---------------------------------------------------------------

def body_force_vector(element, b):  #Vector local de fuerzas de cuerpo. b = (bx, by) por unidad de volumen.
    nn = element.ref.number_of_nodes()
    fe = numpy.zeros(2 * nn)
    points, weights = element.ref.gauss_rule()
    for (xi, eta), w in zip(points, weights):
        N, dN = element.ref.shape(xi, eta)
        detJ = numpy.linalg.det(element.X.T @ dN)

        for a in range(nn):
            fe[2 * a] += N[a] * b[0] * detJ * element.thickness * w
            fe[2 * a + 1] += N[a] * b[1] * detJ * element.thickness * w
    return fe
    
def assemble_body_force(f, elements, b):
    for element in elements:
        fe = body_force_vector(element, b)
        dofs = element.global_dofs()

        for i in range(len(dofs)):
            f[dofs[i]] += fe[i]


# Presion sobre aristas (presion hidrostatica)
# ---------------------------------------------------------------

def edge_shape(number_of_edge_nodes, s):  #Funciones de forma 1D sobre una arista, coord local s en [-1, 1].
    if number_of_edge_nodes == 2:   # Orden: [a, b]
        Na = 0.5 * (1 - s)
        Nb = 0.5 * (1 + s)
        N = numpy.array([Na, Nb])
        dNa = -0.5
        dNb = 0.5
        dN = numpy.array([dNa, dNb])
    else:  # Orden: [a, b, (lado)]
        Na = 0.5 * s * (s - 1)
        Nb = 0.5 * s * (s + 1)
        Nlado = 1 - s * s
        N = numpy.array([Na, Nb, Nlado])
        dNa = s - 0.5
        dNb = s + 0.5
        dNlado = -2 * s
        dN = numpy.array([dNa, dNb, dNlado])
    return N, dN

def pressure_vector(element, edge, pressure):   # Vector local de una arista cargada con presion normal.
    # pressure(x, y) > 0 comprime: la traccion es -p * n, con n la normal exterior.
    nn = element.ref.number_of_nodes()
    fe = numpy.zeros(2 * nn)

    s_points, s_weights = line_rule(3)
    for s, w in zip(s_points, s_weights):
        N, dN = edge_shape(len(edge), s)

        x = numpy.zeros(2)
        tangent = numpy.zeros(2)
        for k, a in enumerate(edge):
            x += N[k] * element.X[a]
            tangent += dN[k] * element.X[a]

        length = numpy.linalg.norm(tangent)
        normal = numpy.array([tangent[1], -tangent[0]]) / length   # exterior (antihorario)
        p = pressure(x[0], x[1])

        for k, a in enumerate(edge):
            fe[2 * a] += N[k] * (-p * normal[0]) * length * element.thickness * w
            fe[2 * a + 1] += N[k] * (-p * normal[1]) * length * element.thickness * w

    return fe

def assemble_pressure(f, elements, on_edge, pressure): # Suma la presion en todas las aristas donde on_edge(p, q) sea verdadero (p y q: coordenadas de las dos esquinas de la arista).
    for element in elements:
        for edge in EDGES[element.ref.name]:
            p = element.X[edge[0]]
            q = element.X[edge[1]]

            if on_edge(p, q):
                fe = pressure_vector(element, edge, pressure)
                dofs = element.global_dofs()

                for i in range(len(dofs)):
                    f[dofs[i]] += fe[i]
