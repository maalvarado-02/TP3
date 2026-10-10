import numpy
from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.solid2d import Solid2D
from source.systems.dense_matrix import DenseMatrix
from source.output.paraview import write_results

THICKNESS = 1.0

# Elemento a probar:
ELEMENT = "T6"          # "Q4", "T3", "T6" o "Q8"

# --- Leer malla ----------------------------------------------------------
mesh = Mesh()
mesh.read("meshes/patchtest_%s.mesh" % ELEMENT)
mesh.dofs_per_node = Solid2D.dofs_per_node

# --- Material ------------------------------------------------------------
E = 200.0e9
nu = 0.2
material = LinearElastic(E, nu)

# --- Crear elementos -----------------------------------------------------
elements = []
for e in range(mesh.number_of_elements()):
    elements.append(
        Solid2D(
            mesh.element_global_ids[e],
            mesh.connectivity[e],
            mesh,
            material,
            THICKNESS,
            mesh.element_types[e]
        )
    )

# --- Ensamble global -----------------------------------------------------
n = mesh.number_of_dofs()
K = DenseMatrix(n)
for el in elements:
    K.scatter_local_to_global(el)


def dofs(node):
    return mesh.node_global_dofs(node - 1)

# Las cargas de la figura suponen nodos equiespaciados en los bordes izq y der
def y(node):
    return mesh.node(node - 1).y

H = y(4) - y(1)
for a, b, c in [(1, 4, 7), (3, 6, 9)]:
    if abs((y(b) - y(a)) - (y(c) - y(b))) > 1e-12:
        raise ValueError("Bordes laterales deben tener nodos equiespaciados")

# --- Vector de fuerzas nodales (work-equivalent) -------------------------
f = numpy.zeros(n)
F = 1.0e6

LOADS = {
    "Q4": {3: 1, 6: 2, 9: 1, 7: -1},
    "T3": {3: 1, 6: 2, 9: 1, 7: -1},
    "T6": {3: 1/3, 16: 4/3, 6: 2/3, 23: 4/3, 9: 1/3,
           14: -4/3, 22: -4/3, 7: -1/3},
    "Q8": {3: 1/3, 15: 4/3, 6: 2/3, 20: 4/3, 9: 1/3,
           13: -4/3, 19: -4/3, 7: -1/3},
}

for node, factor in LOADS[ELEMENT].items():
    f[dofs(node)[0]] += factor * F

# --- Condiciones de borde  ---------------------------------------
fixed = []
values = []

# nodo 1 fijo: ux = 0, uy = 0
fixed += dofs(1)
values += [0.0, 0.0]

# nodo 4 desliza en y: ux = 0
fixed.append(dofs(4)[0])
values.append(0.0)

print("\nelemento: %s | nodos: %d | elementos: %d" % (ELEMENT, n // 2, len(elements)))

# --- Resolver sistema reducido ------------------------------------------
free = [d for d in range(n) if d not in fixed]

Kff = K.values[numpy.ix_(free, free)]
ev = numpy.linalg.eigvalsh(Kff)
print("\nautovalor minimo / maximo de K reducida: %.2e (debe ser > 0)" % (ev.min() / ev.max()))

u = numpy.zeros(n)
u[fixed] = values

u[free] = numpy.linalg.solve(
    Kff,
    f[free] - K.values[numpy.ix_(free, fixed)] @ u[fixed]
)

# --- Reacciones: R = K u - f -------------------------------------------------
R = K.values @ u - f
r = 1/3 if ELEMENT in ("T6", "Q8") else 1.0
print("reaccion x nodo 1: %.4e (esperada %.4e)" % (R[dofs(1)[0]], -r * F))
print("reaccion x nodo 4: %.4e (esperada %.4e)" % (R[dofs(4)[0]], -2 * r * F))


# --- Tension sigma_x en esquinas y puntos de Gauss ---------------------------
def sigma_x(el, xi, eta):
    B, detJ = el.B_and_detJ(xi, eta)
    ue = numpy.array([u[d] for d in el.global_dofs()])
    return (el.D @ (B @ ue))[0]


sigma_exact = 2 * F / (H * THICKNESS)

print("\nsigma_x exacta = 2F/(H t) = %.6e Pa" % sigma_exact)
print(" elem        esquinas (min .. max)                 Gauss (min .. max)")
corner_all = []
gauss_all = []
for el in elements:
    corner = [sigma_x(el, xi, eta) for xi, eta in el.ref.node_coords]
    gauss = [sigma_x(el, xi, eta) for xi, eta in el.ref.gauss_rule()[0]]
    corner_all += corner
    gauss_all += gauss
    print("  %d     %.9e .. %.9e   %.9e .. %.9e"
          % (el.global_id, min(corner), max(corner), min(gauss), max(gauss)))

TOL = 1.0e-9
values_x = numpy.array(corner_all + gauss_all)
spread = (values_x.max() - values_x.min()) / sigma_exact
error = abs(values_x - sigma_exact).max() / sigma_exact
print("\ndispersion relativa de sigma_x en el parche: %.2e" % spread)
print("error relativo respecto de 2F/(H t):          %.2e" % error)
print("sigma_x constante y exacta:",
      "PASA" if (error < TOL and ev.min() > 0.0) else "FALLA")

# --- Exportar resultados -------------------------------------------------
write_results("results/patch%s_sigma_x.vtu" % ELEMENT, mesh, elements, u)
