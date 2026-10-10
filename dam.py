import numpy

from source.elements.shape_functions import REFERENCE_ELEMENTS, reference_element
from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.solid2d import Solid2D
from source.systems.dense_matrix import DenseMatrix
# from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results


# ETAPA 1: elementos de referencia
# ---------------------------------------------------------------
for name in REFERENCE_ELEMENTS:
    ref = reference_element(name)
    points, weights = ref.gauss_rule()

    print(name, "| nodos:", ref.number_of_nodes(),
          "| puntos de Gauss:", len(weights),
          "| meshio:", ref.meshio_type)


# ETAPA 2: material con matriz constitutiva D
# ---------------------------------------------------------------
material = LinearElastic(E=21.0e9, nu=0.2, rho=2400.0)
print(material.D_plane_stress())


# ETAPA 3: malla y elementos
# ---------------------------------------------------------------
mesh = Mesh()
mesh.read("meshes/patchtest.mesh")
mesh.dofs_per_node = Solid2D.dofs_per_node

elements = []
for e in range(mesh.number_of_elements()):
    element = Solid2D(
        mesh.element_global_ids[e],
        mesh.connectivity[e],
        mesh,
        material,
        1.0,                          # espesor unitario (2D)
        mesh.element_types[e],
    )
    elements.append(element)


# ETAPA 4: ensamble de K
# ---------------------------------------------------------------
# number_of_dofs = mesh.number_of_dofs()
# K = DenseMatrix(number_of_dofs)
# for element in elements:
#     K.scatter_local_to_global(element)


# ETAPA 5: cargas (peso propio y presion hidrostatica)
# ---------------------------------------------------------------
# f = numpy.zeros(number_of_dofs)
# ... ensamble de fuerzas de cuerpo y de arista ...
# K0 = copia de K y f0 = copia de f (antes de aplicar CB, para reacciones)


# ETAPA 6: condiciones de borde en la base y solucion
# ---------------------------------------------------------------
# for node in base_nodes:
#     for dof in mesh.node_global_dofs(node):
#         apply_dirichlet_bc(K, f, dof, 0.0)
# u = solve(K, f)


# ETAPA 7: postproceso y resultados
# ---------------------------------------------------------------
# reacciones, tensiones, momentos, factores de seguridad
# write_results("results/dam.vtu", mesh, elements, u)
