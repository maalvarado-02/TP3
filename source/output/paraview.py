import os
import meshio
import numpy


def quad_cells(mesh):
    return [("quad", [list(map(int, conn)) for conn in mesh.connectivity])]

def element_stress(element, u):
    ue = u[element.global_dofs()]
    stresses = []

    for (xi, eta), w in zip(*element.ref.gauss_rule()):
        B, _ = element.B_and_detJ(xi, eta)
        sigma = element.D @ (B @ ue)   # [σxx, σyy, τxy]
        stresses.append(sigma)

    return sum(stresses) / len(stresses)


def write(filename, points, cells, displacement, stress):
    directory = os.path.dirname(filename)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={"displacement": displacement},
        cell_data={"stress": [stress]},
    )

    output_mesh.write(filename)


def write_results(filename, mesh, elements, u):
    points = []
    displacement = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        points.append([node.x, node.y, 0.0])
        displacement.append([u[dofs[0]], u[dofs[1]], 0.0])

    stress = []
    for element in elements:
        stress.append(element_stress(element, u))

    write(filename, points, quad_cells(mesh), displacement, stress)
