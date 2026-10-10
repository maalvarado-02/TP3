import numpy

from source.elements.shape_functions import reference_element

# Tipos habilitados por ahora. Al agregar un elemento, sumarlo aca.
SUPPORTED = ("QUAD4",)


class Solid2D: 
    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, thickness,
                 element_type):
        if element_type not in SUPPORTED:
            raise ValueError("Element type not supported yet: " + element_type)

        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness
        self.ref = reference_element(element_type)

        if len(node_global_ids) != self.ref.number_of_nodes():
            raise ValueError("Wrong number of nodes for " + element_type)

        # coordenadas nodales (nn, 2)
        self.X = numpy.zeros((self.ref.number_of_nodes(), 2))
        for a, node_global_id in enumerate(node_global_ids):
            node = mesh.node(node_global_id)
            self.X[a, 0] = node.x
            self.X[a, 1] = node.y

        self.D = material.D_plane_stress()

        # Ke = sum_g B^T D B detJ t w
        points, weights = self.ref.gauss_rule()
        nd = self.dofs_per_node * self.ref.number_of_nodes()
        self.Ke = numpy.zeros((nd, nd))

        for (xi, eta), w in zip(points, weights):
            B, detJ = self.B_and_detJ(xi, eta)
            self.Ke += B.T @ self.D @ B * detJ * thickness * w

    def B_and_detJ(self, xi, eta):
        """Matriz B (3 x 2nn) y det(J) en el punto (xi, eta)."""
        N, dN = self.ref.shape(xi, eta)

        J = self.X.T @ dN                      # J[i, j] = dx_i / dxi_j
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

    def global_dofs(self):
        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def stiffness_matrix(self):
        return self.Ke