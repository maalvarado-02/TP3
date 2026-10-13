import numpy

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.solid2d import Solid2D
from source.systems.dense_matrix import DenseMatrix
from source.systems.loads import assemble_gravity, assemble_pressure
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results


# --- Elemento de la malla  -----------------------------------------------
ELEMENT = "Q4"          # "Q4", "T3", "T6" o "Q8"

# ETAPA 1: Datos de geometria
# ---------------------------------------------------------------
G = 9.81
RHO_WATER = 1000.0
 
HEIGHT = 30.0
BASE = 18.0
CREST = 3.0
FREEBOARD = 2.0
WATER = HEIGHT - FREEBOARD      # nivel del agua medido desde la base
TOL = 1.0e-9

# ETAPA 2: Material con matriz constitutiva D (plane stress)
# ---------------------------------------------------------------


# ETAPA 3: Malla y elementos
# ---------------------------------------------------------------
mesh = Mesh()
mesh.read("meshes/dam_%s.mesh" % ELEMENT)
mesh.dofs_per_node = Solid2D.dofs_per_node
n = mesh.number_of_dofs()

elements = []
for e in range(mesh.number_of_elements()):
    element = Solid2D(
        mesh.element_global_ids[e],
        mesh.connectivity[e],
        mesh,
        mesh.element_material(e),
        mesh.thickness,
        mesh.element_types[e],
        mesh.analysis,
    )
    elements.append(element)

print("\nTipo de Elemento: %s | Nodos: %d | Elementos: %d" % (ELEMENT, n // 2, len(elements)))


# ETAPA 4: Ensamble global de K
# ---------------------------------------------------------------
K = DenseMatrix(n)
for el in elements:
    K.scatter_local_to_global(el)


# ETAPA 5: Cargas (peso propio y presion hidrostatica)
# ---------------------------------------------------------------
f = numpy.zeros(n)

#Peso propio: b=(0, -rho g) con la rho del material de cada elemento
assemble_gravity(f, elements, G)
# Presion hidrostatica sobre la cara aguas arriba (x = 0), solo bajo el nivel del agua

def on_upstream_face(p, q):
    return (abs(p[0]) < TOL and abs(q[0]) < TOL
            and p[1] <= WATER + TOL and q[1] <= WATER + TOL)
 
def hydrostatic_pressure(x, y):
    return RHO_WATER * G * (WATER - y)
 
assemble_pressure(f, elements, on_upstream_face, hydrostatic_pressure)

# Copias antes de aplicar CB (para autovalores y reacciones)
K0 = K.values.copy()
f0 = f.copy()


# ETAPA 6: Condiciones de borde en la base
# ---------------------------------------------------------------
fixed = []
values = []

base_nodes = [k for k in range(n // Solid2D.dofs_per_node) if abs(mesh.node(k).y) < TOL]

for node in base_nodes:
    for dof in mesh.node_global_dofs(node):
        fixed.append(dof)
        apply_dirichlet_bc(K, f, dof, 0.0)


# ETAPA 7: Resolver sistema (reducido)
# ---------------------------------------------------------------
u = solve(K, f)

# --- Autovalores de K reducida (chequeo de estabilidad) ------------------
free = [d for d in range(n) if d not in fixed]
ev = numpy.linalg.eigvalsh(K0[numpy.ix_(free, free)])
print("\nAutovalor minimo/maximo de K reducida: %.2e (debe ser > 0)" % (ev.min() / ev.max()))


# ETAPA 8: Postproceso y resultados (reacciones, tensiones, momentos, FS)
# ---------------------------------------------------------------
write_results("results/dam_%s.vtu" % ELEMENT, mesh, elements, u)
