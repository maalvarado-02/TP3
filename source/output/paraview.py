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

def principal_max(sigma):
    #tension principal maxima en el plano 
    sx, sy, txy = sigma
    return (sx + sy)/ 2 + numpy.sqrt(((sx-sy)/2)**2 + txy**2)
def element_sigma1_max(element, u):
    # maximo de sigma_1 entre los puntos de gauss del elemento
    ue=u[element.global_dofs()]
    values=[]
    for xi, eta in element.ref.gauss_rule()[0]:
        B,_=element.B_and_detJ(xi, eta)
        values.append(principal_max(element.D @ (B @ ue)))
    return max(values)
    

def write(filename, points, cells, displacement, stress_blocks, sigma1_blocks):
    directory = os.path.dirname(filename)
    if directory != "":
        os.makedirs(directory, exist_ok=True)

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={"displacement": displacement},
        cell_data={"stress": stress_blocks, "sigma_1": sigma1_blocks},
    )

    output_mesh.write(filename)


def write_results(filename, mesh, elements, u, scale=0.0):
    #scale: factor de escala de la deformada
    points = []
    displacement = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        points.append([node.x + scale*u[dofs[0]], node.y + scale*u[dofs[1]], 0.0])
        displacement.append([u[dofs[0]], u[dofs[1]], 0.0])
    cells, block_indices= cell_blocks(elements)

    stress_blocks = []
    sigma1_blocks=[]
    for indices in block_indices:
        stress_blocks.append([element_stress(elements[i], u) for i in indices])
        sigma1_blocks.append([element_sigma1_max(elements[i], u) for i in indices])
    write(filename, points, cells, displacement, stress_blocks, sigma1_blocks)
