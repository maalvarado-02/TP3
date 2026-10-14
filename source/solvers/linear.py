import numpy
from scipy.sparse.linalg import spsolve

def solve(K, f):   # Solver denso (solo para matrices pequeñas)
    return numpy.linalg.solve(K.values, f)

def solve_sparse(K, f):   # Solver disperso (para mallas grandes)
    Kcsc = K.to_csc()
    return spsolve(Kcsc, f)