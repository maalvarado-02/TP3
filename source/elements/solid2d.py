import numpy

from source.elements.shape_functions import reference_element

# Matriz de rigidez elemental (Ke = sum_g B^T D B detJ t wg) de un elemento 2D isoparamétrico (triangular o cuadrilátero)

SUPPORTED = ("TRI3", "TRI6", "QUAD4", "QUAD8") # Tipos habilitados por ahora. Al agregar un elemento, sumarlo aca.

class Solid2D:
    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness, element_type, formulation): # Elemento finito 2D
        if element_type not in SUPPORTED:
            raise ValueError("Tipo de elemento no soportado aún: " + element_type)

        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness
        self.ref = reference_element(element_type)
        self.formulation = formulation
        if len(node_global_ids) != self.ref.number_of_nodes():
            raise ValueError("Número incorrecto de nodos para " + element_type)

        # Coordenadas nodales (nn, 2)
        self.X = numpy.zeros((self.ref.number_of_nodes(), 2))
        for a, node_global_id in enumerate(node_global_ids):
            node = mesh.node(node_global_id)
            self.X[a, 0] = node.x
            self.X[a, 1] = node.y

        # Selección de la matriz constitutiva según la formulación
        if formulation == "plane_stress":
            self.D = material.D_plane_stress()
        elif formulation == "plane_strain":
            self.D = material.D_plane_strain()
        else:
            raise ValueError("Formulacion desconocida: " + formulation)

        # Ke = sum_g B^T D B detJ t wg
        points, weights = self.ref.gauss_rule()
        nd = self.dofs_per_node * self.ref.number_of_nodes()
        self.Ke = numpy.zeros((nd, nd))

        for (xi, eta), w in zip(points, weights):   # Arma la matriz de rigidez elemental Ke del elemento
            B, detJ = self.B_and_detJ(xi, eta)
            self.Ke += B.T @ self.D @ B * detJ * thickness * w

    def B_and_detJ(self, xi, eta):   # Calcula la matriz de deformaciones B (3 x 2nn) y el det(J) en un punto (xi, eta) del elemento.
        N, dN = self.ref.shape(xi, eta)

        J = self.X.T @ dN                   # J[i, j] = dx_i / dxi_j
        detJ = numpy.linalg.det(J)

        if detJ <= 0.0:
            raise ValueError("Element %d: detJ <= 0 (%g). Revisar orden de nodos."
                             % (self.global_id, detJ))

        dN_dx = dN @ numpy.linalg.inv(J)       # (nn, 2): [dN/dx, dN/dy]

        nn = self.ref.number_of_nodes()
        B = numpy.zeros((3, 2 * nn))
        for a in range(nn):
            B[0, 2 * a] = dN_dx[a, 0]
            B[1, 2 * a + 1] = dN_dx[a, 1]
            B[2, 2 * a] = dN_dx[a, 1]
            B[2, 2 * a + 1] = dN_dx[a, 0]

        return B, detJ

    def global_dofs(self):   # Lista de grados de libertad globales del elemento (2 por nodo)
        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def stiffness_matrix(self):   # Devuelve la matriz de rigidez elemental (Ke) del elemento
        return self.Ke

    def stress_at(self, xi, eta, u):   # Tension [sxx, syy, txy] en un punto (xi, eta) del elemento
        ue = u[self.global_dofs()]
        B, _ = self.B_and_detJ(xi, eta)
        return self.D @ (B @ ue)

    def stress_at_gauss(self, u):   # Tension en cada punto de Gauss
        points, _ = self.ref.gauss_rule()
        return [self.stress_at(xi, eta, u) for xi, eta in points]

    def average_stress(self, u):   # Tension promedio del elemento en los puntos de Gauss (para el calculo)
        stresses = self.stress_at_gauss(u)
        return sum(stresses) / len(stresses)

    def principal_stress(self, u):   # Tensiones principales del elemento (s1 >= s2 >= s3) con szz incluida
        sxx, syy, txy = self.average_stress(u)
        szz = self.sigma_zz(sxx, syy)
        s1 = 0.5*(sxx + syy) + numpy.sqrt(0.25*(sxx - syy)**2 + txy**2)
        s2 = 0.5*(sxx + syy) - numpy.sqrt(0.25*(sxx - syy)**2 + txy**2)
        return numpy.array([s1, s2, szz])

    def vonMises_stress(self, u):   # Tensión equivalente de Von Mises del elemento
        sxx, syy, txy = self.average_stress(u)
        szz = self.sigma_zz(sxx, syy)
        s_vm = numpy.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) + 3 * txy**2)
        return s_vm

    def sigma_zz(self, sxx, syy): # plane_strain: szz = nu (sxx + syy);  plane_stress: szz = 0
        if self.formulation == "plane_strain":
            return self.material.nu * (sxx + syy)
        return 0.0