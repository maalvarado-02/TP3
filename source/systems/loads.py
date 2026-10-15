import numpy

# Creacion de vectores de carga globales de fuerzas de cuerpo y superficie

# Fuerzas de cuerpo (peso propio)
# ---------------------------------------------------------------

# Vector local de fuerzas de cuerpo (fe = ∫ N^T b dV) -- b = (bx, by) fuerza x unidad de volumen
def body_force_vector(element, b):
    nn = element.ref.number_of_nodes()
    fe = numpy.zeros(2 * nn)
    points, weights = element.ref.gauss_rule() # Gauss del elemento
    for (xi, eta), w in zip(points, weights):
        N, dN = element.ref.shape(xi, eta)
        detJ = numpy.linalg.det(element.X.T @ dN)

        for a in range(nn):   # Ensamblado local: fe[a] += N[a] * b * detJ * t * w
            fe[2 * a] += N[a] * b[0] * detJ * element.thickness * w
            fe[2 * a + 1] += N[a] * b[1] * detJ * element.thickness * w
    return fe
 
# Ensambla el vector global de fuerzas de cuerpo    
def assemble_body_force(f, elements, b):
    for element in elements:
        fe = body_force_vector(element, b)
        dofs = element.global_dofs()

        for i in range(len(dofs)):
            f[dofs[i]] += fe[i]


# Fuerzas de superficie (presion hidrostatica sobre aristas)
# ---------------------------------------------------------------

# Vector local de una arista cargada con presion normal fe = ∫ N^T (-p n) ds - la traccion es -p * n (n la normal exterior)
def pressure_vector(element, edge, pressure):
    nn = element.ref.number_of_nodes()
    fe = numpy.zeros(2 * nn)
    line = element.ref.edge_element()
    s_points, s_weights = line.gauss_rule()   # Gauss 1D sobre la arista
    for s, w in zip(s_points, s_weights):
        N, dN = line.shape(s)

        x = numpy.zeros(2)
        tangent = numpy.zeros(2)
        for k, a in enumerate(edge):   # Posicion y tangente
            x += N[k] * element.X[a]
            tangent += dN[k] * element.X[a]

        length = numpy.linalg.norm(tangent)   # Longitud ds
        normal = numpy.array([tangent[1], -tangent[0]]) / length   # Normal exterior n (antihorario)
        p = pressure(x[0], x[1])   # Tracción aplicada p(x,y)

        for k, a in enumerate(edge):   # Ensamblado local: fe[a] += N[a] * (-p n) * ds * t * w
            fe[2 * a] += N[k] * (-p * normal[0]) * length * element.thickness * w
            fe[2 * a + 1] += N[k] * (-p * normal[1]) * length * element.thickness * w

    return fe

# Ensambla el vector global de presion en todas las aristas donde on_edge(p, q) sea verdadero (p y q: coordenadas de los dos nodos de la arista).
def assemble_pressure(f, elements, on_edge, pressure):
    for element in elements:
        for edge in element.ref.edges:
            p = element.X[edge[0]]
            q = element.X[edge[1]]

            if on_edge(p, q):
                fe = pressure_vector(element, edge, pressure)
                dofs = element.global_dofs()
                for i in range(len(dofs)):
                    f[dofs[i]] += fe[i]
