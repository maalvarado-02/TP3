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
def assemble_gravity(f, elements, g):
    for element in elements:
        b=(0.0, -element.material.rho*g)
        fe= body_force_vector(element, b)
        dofs= element.global_dofs()
        for i in range(len(dofs)):
            f[dofs[i]] += fe[i]

def edge_pressure_vector (element, edge, pressure, n_gauss=3):
    nn=element.ref.number_of_nodes()
    fe=numpy.zeros(2*nn)
    X= element.X[list(edge)]
    t_points, t_weights=numpy.polynomial.legendre.leggauss(n_gauss)
    for t, w in zip(t_points, t_weights):
        if len(edge)==2:
            N=numpy.array([(1-t)/2, (1+t)/2])
            dN= numpy.array([-0.5,0.5])
        else:
            N= numpy.array([t*(t-1)/2, t*(t+1)/2, 1- t**2])
            dN=numpy.array([t-0.5, t+0.5, -2*t])

        x, y= N @ X
        dx, dy= dN @ X
        p= pressure(x,y)
        tx= -p*dy
        ty=p*dx
        for k, a in enumerate(edge):
            fe[2*a] += N[k]*tx*element.thickness*w
            fe[2*a+1] += N[k]*ty*element.thickness*w
    return fe

def assemble_pressure(f, elements, on_edge, pressure):
    for element in elements:
        for edge in EDGES[element.ref.name]:
            p=element.X[edge[0]]
            q=element.X[edge[1]]
            if not on_edge(p,q):
                continue

            fe=edge_pressure_vector(element, edge, pressure)
            dofs=element.global_dofs()
            for i in range(len(dofs)):
                f[dofs[i]] += fe[i]
                
