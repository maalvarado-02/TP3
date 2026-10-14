import numpy

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.solid2d import Solid2D
from source.systems.dense_matrix import DenseMatrix
from source.systems.loads import assemble_gravity, assemble_pressure
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results, element_sigma1_max


# --- Elemento de la malla  -----------------------------------------------
ELEMENT = "Q4"          # "Q4", "T3", "T6" o "Q8"

# --- Tipo de formulación constitutiva ------------------------------------
FORMULATION = "plane_strain"     # "plane_stress" o "plane_strain"

# --- Factor de escala de la deformada para ParaView/graficos -------------
 # scale = 0 (geometría original sin deformación)
 # scale = 1 (deformada real (x + u))
 # scale > 1 (la deformada se exagera para visualizarla mejor)
SCALE = 100

# ETAPA 1: Datos de geometria
# ---------------------------------------------------------------
THICKNESS = 1.0           # Espesor unitario (2D): cargas por metro de ancho
G = 9.81
rhoAgua = 1000      # Densidad del agua (kg/m3)
 
H = 30   # Altura de la presa
b = 18   # Ancho de la base
c = 3    # Ancho de la cresta
d = 2    # Altura libre sobre el nivel del agua
Hagua = H - d   # Nivel del agua medido desde la base

TOL = 1.0e-9

# ETAPA 2: Material con matriz constitutiva D - Hormigon H°21 
# ---------------------------------------------------------------
E = 21.0e9            # Modulo de elasticidad (Pa)
nu = 0.2              # Coef de Poisson
rhoHormigon = 2400  # Densidad (kg/m3)

material = LinearElastic(E, nu, rhoHormigon)


# ETAPA 3: Malla y elementos
# ---------------------------------------------------------------
mesh = Mesh()
mesh.read("meshes/dam_%s.mesh" % ELEMENT)
mesh.dofs_per_node = Solid2D.dofs_per_node

n_dofs = mesh.number_of_dofs()
n_nodes = mesh.number_of_nodes()

elements = []
for e in range(mesh.number_of_elements()):
    element = Solid2D(
        mesh.element_global_ids[e],
        mesh.connectivity[e],
        mesh,
        mesh.element_material(e),
        THICKNESS,
        mesh.element_types[e],
        FORMULATION
    )
    elements.append(element)

print("\nTipo de Elemento: %s | Nodos: %d | Elementos: %d" % (ELEMENT, n_dofs // 2, len(elements)))
print("\nTipo de Formulacion: %s" % FORMULATION)

# ETAPA 4: Ensamble global de K
# ---------------------------------------------------------------
K = DenseMatrix(n_dofs)
for el in elements:
    K.scatter_local_to_global(el)


# ETAPA 5: Cargas (peso propio y presion hidrostatica)
# ---------------------------------------------------------------
f = numpy.zeros(n_dofs)

# Peso propio: fuerza de cuerpo x unidad de volumen b=(0, -rho.g)
assemble_body_force(f, elements, (0.0, -rhoHormigon * G))
"""
# Presion hidrostatica sobre la cara aguas arriba (x = 0), solo bajo el nivel del agua

def on_upstream_face(p, q):
    return (abs(p[0]) < TOL and abs(q[0]) < TOL
            and p[1] <= Hagua + TOL and q[1] <= Hagua + TOL)
 
def hydrostatic_pressure(x, y):
    return rhoAgua * G * (Hagua - y)
 
assemble_pressure(f, elements, on_upstream_face, hydrostatic_pressure)
"""
# Copias antes de aplicar CB (para autovalores y reacciones)
K0 = K.values.copy()
f0 = f.copy()


# ETAPA 6: Condiciones de borde en la base
# ---------------------------------------------------------------
fixed = []

base_nodes = [i for i in range(n_nodes) if abs(mesh.node(i).y) < TOL]

for node in base_nodes:
    for dof in mesh.node_global_dofs(node):
        fixed.append(dof)
        fixed.append(dof)
        apply_dirichlet_bc(K, f, dof, 0.0)


# ETAPA 7: Resolver sistema (reducido)
# ---------------------------------------------------------------
u = solve(K, f)

# --- Autovalores de K reducida (chequeo de estabilidad) ------------------
free = [d for d in range(n_dofs) if d not in fixed]
ev = numpy.linalg.eigvalsh(K0[numpy.ix_(free, free)])
print("\nAutovalor minimo/maximo de K reducida: %.2e (debe ser > 0)" % (ev.min() / ev.max()))
if ev.min() <= 0:
    print("Advertencia: matriz K reducida no es definida positiva")

# --- Reacciones en la base: R = K0 u - f0 -------------------------------------------
R = K0 @ u - f0


# ETAPA 8: Postproceso y resultados (reacciones, tensiones, momentos, FS)
# ---------------------------------------------------------------
 
# --- Momento estabilizante -----------------------------
area = 0.5 * (b+c) * H
W = rhoHormigon * G * area * THICKNESS
x_cg = (b**2+b*c+c**2) / (3*(b + c))  # Desde la cara aguas arriba
M_estabilizante = W * (b - x_cg)      # El peso gira antihorario respecto del pie aguas abajo (b, 0)

# --- Momento desestabilizante -----------------------------
Fempuje = 0.5 * rhoAgua * G * Hagua**2 * THICKNESS
Hempuje = Hagua / 3                          # Brazo de palanca del empuje (centroide de la presión)
M_desestabilizante = Fempuje * Hempuje       # El empuje gira horario (se toma en modulo)

FS = M_estabilizante / M_desestabilizante

# --- Momento de las reacciones de la base (FEM); Rx no aporta porque esta en y = 0
M_reactions = 0.0
for node in base_nodes:
    M_reactions += (mesh.node(node).x - b) * R[2 * node + 1]

# --- Reacciones totales en la base (FEM) -----------------------------
Rx = sum(R[2*node] for node in base_nodes)
Ry = sum(R[2*node+1] for node in base_nodes)

print("\n--- Equilibrio global (por metro de ancho) ---")
print("  Peso W        = %.4f MN   | Ry (FEM) = %.4f MN" % (W / 1e6, Ry / 1e6))
print("  Empuje F      = %.4f MN   | Rx (FEM) = %.4f MN" % (Fempuje / 1e6, -Rx / 1e6))
print("  M Estabilizante   W*(B-x_cg)    = %.6e N m" % M_estabilizante)
print("  M Desestabilizante  F*Hagua/3   = %.6e N m" % M_desestabilizante)
print("  Factor de seguridad al vuelco   = %.3f" % FS)
print("  M Neto (estab - desestab)       = %.6e N m" % (M_estabilizante - M_desestabilizante))
print("  M Momento de las reacciones FEM = %.6e N m (debe ser el neto, con signo contrario)" % M_reactions)

# --- Tensión principal máxima (FEM) ---------------------------------------
sigma1 = []
sigma2 = []
sigma_vm = []

for el in elements:
    sigma1.append(el.principal_stress(u)[0])
    sigma2.append(el.principal_stress(u)[1])
    sigma_vm.append(el.vonMises_stress(u) )

e_max = int(numpy.argmax(sigma1))         # índice del elemento crítico
xc, yc = elements[e_max].X.mean(axis=0)   # Coordenadas del centroide del elemento crítico

print("\n--- Tensión principal máxima ---")
print("  sigma_1 max = %.3f MPa" % (max(sigma1) / 1e6))
print("  sigma_2 min = %.3f MPa" % (min(sigma2) / 1e6))
print("  Von Mises max = %.3f MPa" % (max(sigma_vm) / 1e6))
print("  Elemento crítico: %d" % (e_max + 1))
print("  Ubicación: x = %.2f m, y = %.2f m" % (xc, yc))

# --- Desplazamiento máximo -------------------------------------------------
umax = numpy.abs(u).max()
print("  Desplazamiento máximo = %.3f mm" % (1e3 * umax))

write_results("results/dam_%s.vtu" % ELEMENT, mesh, elements, u, SCALE)   # Arma el archivo VTU para ParaView con la deformada y tensiones