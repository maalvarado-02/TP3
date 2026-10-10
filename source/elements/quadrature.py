import numpy


def triangle_rule(order):
    if order == 1:
        points = numpy.array([[1 / 3, 1 / 3]])
        weights = numpy.array([0.5])
    elif order == 3:
        points = numpy.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        weights = numpy.array([1 / 6, 1 / 6, 1 / 6])
    else:
        raise ValueError("Triangle rule not available")
    return points, weights


def quad_rule(n):
    x, w = numpy.polynomial.legendre.leggauss(n)
    points = []
    weights = []
    for j in range(n):
        for i in range(n):
            points.append([x[i], x[j]])
            weights.append(w[i] * w[j])
    return numpy.array(points), numpy.array(weights)