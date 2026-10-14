import os
import meshio
import numpy


def cell_blocks(elements):   # Agrupa los elementos por tipo de celda (TRI3, TRI6, QUAD4, QUAD8)
    blocks={}
    for i, element in enumerate(elements):
        cell_type = element.ref.meshio_type
        blocks.setdefault(cell_type, []).append(i)

    cells=[]
    for cell_type, indices in blocks.items():
        conn = [list(map(int, elements[i].node_global_ids)) for i in indices]
        cells.append((cell_type, conn))

    return cells, list(blocks.values())


def write(filename, points, cells, point_data, cell_data):   # Escribe el archivo VTU usando meshio
    directory = os.path.dirname(filename)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,)

    output_mesh.write(filename)


def write_results(filename, mesh, elements, u, SCALE):   # Función principal de postproceso
    points = []
    displacement = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        ux = u[dofs[0]]
        uy = u[dofs[1]]
        points.append([node.x+SCALE*ux, node.y+SCALE*uy, 0.0])
        displacement.append([ux, uy, 0.0])
    cells, block_indices = cell_blocks(elements)

    average_stress = [el.average_stress(u) for el in elements]
    principal_stress = [el.principal_stress(u) for el in elements]
    vonMises_stress = [el.vonMises_stress(u) for el in elements]

    stress_blocks = []
    principal_blocks = []
    vonMises_blocks = []

    for indices in block_indices:
        stress_blocks.append([average_stress[i] for i in indices])
        principal_blocks.append([principal_stress[i] for i in indices])
        vonMises_blocks.append([vonMises_stress[i] for i in indices])

    point_data = {
            "displacement": displacement,}
    cell_data = {
        "stress": stress_blocks,
        "principal_stress": principal_blocks,
        "von_mises": vonMises_blocks,}
    
    write(filename, points, cells, point_data=point_data, cell_data=cell_data)

