import numpy

# Aplica las condiciones de borde (Dirichlet o Robin) a los nodos que se pasan

# Condicion de borde Dirichlet (establece directamente el valor de un desplazamiento en un punto o borde)
# ----------------------------------------------------------------------------------------
def apply_dirichlet_bc(K, f, dof, value): 
    if hasattr(K.values, "tolil"):   # Si es sparse, convertir a LIL para edición eficiente
        K.values = K.values.tolil()

    for i in range(K.number_of_dofs):
        f[i] -= K.values[i, dof] * value

    K.values[dof, :] = 0.0    # Anular fila del DOF
    K.values[:, dof] = 0.0    # Anular columna del DOF
    K.values[dof, dof] = 1.0  # Poner 1 en la diagonal
    f[dof] = value   # Ajustar vector de cargas


# Condicion de borde Robin (resortes distribuidos tipo Winkler sobre aristas)
# ----------------------------------------------------------------------------------------
# En el borde: t = -k u  (la fundacion devuelve una traccion proporcional al desplazamiento)
# Limites: k -> 0 es borde libre y k -> infinito es empotramiento (Dirichlet u = 0)

def robin_edge_matrix(element, edge, k_tangential):   # Matriz local de resortes de una arista
    nn = element.ref.number_of_nodes()
    Ke = numpy.zeros((2 * nn, 2 * nn))

    line = element.ref.edge_element()
    s_points, s_weights = line.gauss_rule()
    
    for s, w in zip(s_points, s_weights):
        N, dN = line.shape(s)

        tangent = numpy.zeros(2)
        for k, a in enumerate(edge):
            tangent += dN[k] * element.X[a]
        length = numpy.linalg.norm(tangent)
        t = tangent / length                  # Versor tangente
        k_matrix = k_tangential * numpy.outer(t, t)   # 2x2 en ejes globales (solo direccion tangencial)

        for i, a in enumerate(edge):
            for j, b in enumerate(edge):
                Ke[2*a:2*a+2, 2*b:2*b+2] += N[i] * N[j] * k_matrix * length * element.thickness * w

    return Ke


def apply_robin_bc(K, elements, on_edge, k_tangential):   # Suma los resortes a K en las aristas donde on_edge(p, q) sea verdadero
    for element in elements:
        for edge in element.ref.edges:
            p = element.X[edge[0]]
            q = element.X[edge[1]]

            if on_edge(p, q):
                Ke = robin_edge_matrix(element, edge, k_tangential)
                dofs = element.global_dofs()

                for i in range(len(dofs)):
                    for j in range(len(dofs)):
                        K.add(dofs[i], dofs[j], Ke[i][j])    