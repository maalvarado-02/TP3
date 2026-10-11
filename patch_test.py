import numpy

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.solid2d import Solid2D
from source.systems.dense_matrix import DenseMatrix
from source.output.paraview import write_results
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve

THICKNESS = 1.0   # espesor unitario (2D)

# Elemento a probar:
ELEMENT = "Q8"          # "Q4", "T3", "T6" o "Q8"

# --- Material ------------------------------------------------------------
E = 200.0e9
nu = 0.2
material = LinearElastic(E, nu)

# --- Leer malla ----------------------------------------------------------
mesh = Mesh()
mesh.read("meshes/patchtest_%s.mesh" % ELEMENT)
mesh.dofs_per_node = Solid2D.dofs_per_node
n = mesh.number_of_dofs()

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

print("\nTipo de Elemento: %s | Nodos: %d | Elementos: %d" % (ELEMENT, n // 2, len(elements)))

# --- Ensamble global -----------------------------------------------------
K = DenseMatrix(n)
for el in elements:
    K.scatter_local_to_global(el)

# --- Vector de fuerzas nodales (work-equivalent) -------------------------
def dofs(node):
    return mesh.node_global_dofs(node - 1)

f = numpy.zeros(n)
F = 1.0e6

LOADS = {
    "Q4": {3: 1, 6: 2, 9: 1, 7: -1},
    "T3": {3: 1, 6: 2, 9: 1, 7: -1},
    "T6": {3: 1/3, 16: 4/3, 6: 2/3, 23: 4/3, 9: 1/3,
           14: -4/3, 22: -4/3, 7: -1/3},
    "Q8": {3: 1/3, 15: 4/3, 6: 2/3, 20: 4/3, 9: 1/3,
           13: -4/3, 19: -4/3, 7: -1/3},}

for node, factor in LOADS[ELEMENT].items():
    f[dofs(node)[0]] += factor * F

# Las cargas de la figura suponen nodos equiespaciados en los bordes izq y der
def y(node):
    return mesh.node(node - 1).y

H = y(4) - y(1)
for a, b, c in [(1, 4, 7), (3, 6, 9)]:
    if abs((y(b) - y(a)) - (y(c) - y(b))) > 1e-12:
        raise ValueError("Bordes laterales deben tener nodos equiespaciados")

# --- Condiciones de borde  ---------------------------------------
fixed = []

fixed += dofs(1) # nodo 1 fijo: ux = 0, uy = 0
fixed.append(dofs(4)[0]) # nodo 4 desliza en y: ux = 0

# --- Copias sin condiciones de borde (para autovalores y reacciones) ----
K0 = K.values.copy()
f0 = f.copy()

for dof in fixed:
    apply_dirichlet_bc(K, f, dof, 0.0)

u = solve(K, f)

# --- Autovalores de K reducida (chequeo de estabilidad) ------------------
free = [d for d in range(n) if d not in fixed]
ev = numpy.linalg.eigvalsh(K0[numpy.ix_(free, free)])
print("\nAutovalor minimo/maximo de K reducida: %.2e (debe ser > 0)" % (ev.min() / ev.max()))

# --- Reacciones: R = K0 u - f0 -------------------------------------------
R = K0 @ u - f0
r = 1/3 if ELEMENT in ("T6", "Q8") else 1.0
print("\nReaccion x nodo 1: %.4e (esperada %.4e)" % (R[dofs(1)[0]], -r * F))
print("Reaccion x nodo 4: %.4e (esperada %.4e)" % (R[dofs(4)[0]], -2 * r * F))

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