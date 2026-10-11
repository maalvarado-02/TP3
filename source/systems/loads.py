import numpy

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
