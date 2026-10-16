import numpy

# Reglas de cuadratura de Gauss para la integracion numerica (Devuelven los puntos y pesos de Gauss en el elemento de referencia)
# Se usa la funcion numpy.polynomial.legendre.leggauss(n) que genera los puntos y pesos de la cuadratura de Gauss-Legendre de orden n en el intervalo [-1, 1]

def triangle_rule(order):
    if order == 1:     # Regla de 1 punto para triángulos (orden 1)
        points = numpy.array([[1 / 3, 1 / 3]])
        weights = numpy.array([0.5])
    elif order == 3:   # Regla de 3 puntos para triángulos (orden 2)
        points = numpy.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        weights = numpy.array([1 / 6, 1 / 6, 1 / 6])
    else:
        raise ValueError("Triangle rule not available")
    return points, weights

def quad_rule(n):  # Regla de Gauss 2D para cuadriláteros [-1, 1] x [-1, 1]
    x, w = numpy.polynomial.legendre.leggauss(n) # Producto tensorial de la regla 1D de Gauss-Legendre
    points = []
    weights = []
    for j in range(n):
        for i in range(n):
            points.append([x[i], x[j]])
            weights.append(w[i] * w[j])
    return numpy.array(points), numpy.array(weights)

def line_rule(n): # Regla deGauss 1D en [-1, 1], para cargas sobre aristas.
    return numpy.polynomial.legendre.leggauss(n)    # Funcion numpy con regla 1D de Gauss-Legendre