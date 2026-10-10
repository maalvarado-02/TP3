import os
import meshio
import numpy


def cell_blocks(elements):
    blocks={}
    for i, element in enumerate(elements):
        cell_type = element.ref.meshio_type
        blocks.setdefault(cell_type, []).append(i)

    cells=[]
    for cell_type, indices in blocks.items():
        conn = [list(map(int, elements[i].node_global_ids)) for i in indices]
        cells.append((cell_type, conn))
    return cells, list(blocks.values())


def element_stress(element, u):
    ue = u[element.global_dofs()]
    stresses = []

    for (xi, eta), w in zip(*element.ref.gauss_rule()):
        B, _ = element.B_and_detJ(xi, eta)
        sigma = element.D @ (B @ ue)   # [σxx, σyy, τxy]
        stresses.append(sigma)

    return sum(stresses) / len(stresses)


def write(filename, points, cells, displacement, stress_blocks):
    directory = os.path.dirname(filename)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={"displacement": displacement},
        cell_data={"stress": stress_blocks},
    )

    output_mesh.write(filename)


def write_results(filename, mesh, elements, u):
    points = []
    displacement = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        points.append([node.x, node.y, 0.0])
        displacement.append([u[dofs[0]], u[dofs[1]], 0.0])
    cells, block_indices= cell_blocks(elements)

    stress_blocks = []
    for indices in block_indices:
        stress_blocks.append([elemnt_stress(elements[i], u) for i in indices])

    write(filename, points, cells, displacement, stress_blocks)
