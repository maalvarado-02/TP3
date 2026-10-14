import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

# Armado de matrices densas o esparsas

class DenseMatrix:   # Arma matriz densa (solo para matrices pequeñas)
    def __init__(self, number_of_dofs):
        self.number_of_dofs = number_of_dofs
        self.values = np.zeros((number_of_dofs, number_of_dofs))

    def add(self, i, j, value):
        self.values[i, j] += value

    def set(self, i, j, value):
        self.values[i, j] = value

    def scatter_local_to_global(self, element):
        Ke = element.stiffness_matrix()
        dofs = element.global_dofs()

        for a in range(len(dofs)):
            for b in range(len(dofs)):
                self.add(dofs[a], dofs[b], Ke[a, b])


class SparseMatrix:   # Arma matriz sparse (para mallas grandes)
    def __init__(self, number_of_dofs):
        self.number_of_dofs = number_of_dofs
        self.values = lil_matrix((number_of_dofs, number_of_dofs))

    def add(self, i, j, value):
        self.values[i, j] += value

    def set(self, i, j, value):
        self.values[i, j] = value

    def scatter_local_to_global(self, element):
        Ke = element.stiffness_matrix()
        dofs = element.global_dofs()

        for a in range(len(dofs)):
            for b in range(len(dofs)):
                self.values[dofs[a], dofs[b]] += Ke[a, b]

    def to_csc(self):
        return csc_matrix(self.values)
