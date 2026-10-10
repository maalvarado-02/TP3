import numpy

from source.elements.shape_functions import REFERENCE_ELEMENTS, reference_element


for name in REFERENCE_ELEMENTS:
    ref = reference_element(name)
    nn = ref.number_of_nodes()

    print("==", ref.name)

    # 1) N_i(nodo j) = delta_ij
    err = 0.0
    for j, (xi, eta) in enumerate(ref.node_coords):
        N, _ = ref.shape(xi, eta)
        err = max(err, abs(N - numpy.eye(nn)[j]).max())
    print("  delta de Kronecker, error:", err)

    # punto interior de prueba
    if ref.is_triangle:
        xi, eta = 0.2, 0.3
    else:
        xi, eta = 0.3, -0.4

    N, dN = ref.shape(xi, eta)

    # 2) particion de la unidad y suma de derivadas nula
    print("  sum N - 1:", N.sum() - 1.0)
    print("  sum dN:", dN.sum(axis=0))

    # 3) derivadas contra diferencias finitas
    h = 1e-6
    Np, _ = ref.shape(xi + h, eta)
    Nm, _ = ref.shape(xi - h, eta)
    dxi_fd = (Np - Nm) / (2 * h)

    Np, _ = ref.shape(xi, eta + h)
    Nm, _ = ref.shape(xi, eta - h)
    deta_fd = (Np - Nm) / (2 * h)

    print("  derivadas vs dif. finitas:",
          abs(dxi_fd - dN[:, 0]).max(),
          abs(deta_fd - dN[:, 1]).max())

    # 4) reproduccion de campo lineal: sum N_i * x_i = x
    X = numpy.array(ref.node_coords)
    print("  reproduce (xi, eta):", abs(N @ X - numpy.array([xi, eta])).max())

    # 5) cuadratura: suma de pesos = area del elemento de referencia
    points, weights = ref.gauss_rule()
    print("  suma de pesos - area:", weights.sum() - ref.reference_area())
    print("  puntos de Gauss:", len(weights))