import numpy
import tempfile
import io
import os
import contextlib

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.solid2d import Solid2D
from source.systems.matrix import SparseMatrix
from source.systems.loads import assemble_body_force, assemble_pressure
from source.systems.boundary_conditions import apply_dirichlet_bc, apply_robin_bc
from source.solvers.linear import solve_sparse
from make_dam_mesh import make_dam_mesh

rho_w=1000
G=9.81
THICKNESS=1.0
TOL = 1e-9

#Función auxiliar para análisis de convergencia. Devuelve todos los valores necesarios para comparar mallas.
def solve_dam(element="Q4", formulation="plane_strain", base_bc="fixed", nx=10, ny=20, ny_top=4, 
                  H=30.0, b=18.0, c=3.0, d=2.0, t=0.0, KN=1e12, KT=1e12):
    
    # --- Geometría ---
    Hagua = H - d

    # --- Malla ---
    with tempfile.TemporaryDirectory() as folder:
        filename = os.path.join(folder, "dam.mesh")
        with contextlib.redirect_stdout(io.StringIO()):
            make_dam_mesh(element=element, nx=nx, ny=ny, ny_top=ny_top, filename=filename, height=H, 
                          base=b, crest=c, freeboard=d, upstream_batter=t)
        mesh = Mesh()
        mesh.read(filename)

    mesh.dofs_per_node = Solid2D.dofs_per_node

    n_dofs = mesh.number_of_dofs()
    n_nodes = mesh.number_of_nodes()

    # --- Elementos ---
    elements = []
    for e in range(mesh.number_of_elements()):
        el = Solid2D(
            mesh.element_global_ids[e],
            mesh.connectivity[e],
            mesh,
            mesh.element_material(e),
            THICKNESS,
            mesh.element_types[e],
            formulation)
        elements.append(el)

    # --- Matriz global ---
    K = SparseMatrix(n_dofs)
    for el in elements:
        K.scatter_local_to_global(el)

    # --- Cargas ---
    f = numpy.zeros(n_dofs)
    rho_h = elements[0].material.rho

    assemble_body_force(f, elements, (0.0, -rho_h * G))

    def on_upstream_face(p, q):
        return (abs(p[0]) < TOL and abs(q[0]) < TOL
                and p[1] <= Hagua + TOL and q[1] <= Hagua + TOL)

    def hydrostatic_pressure(x, y):
        return rho_w * G * (Hagua - y)

    assemble_pressure(f, elements, on_upstream_face, hydrostatic_pressure)

    # Copias para reacciones
    K0 = K.values.copy()
    f0 = f.copy()

    # --- Condiciones de borde ---
    fixed = set()
    base_nodes = [i for i in range(n_nodes) if abs(mesh.node(i).y) < TOL]

    if base_bc == "fixed":
        for node in base_nodes:
            for dof in mesh.node_global_dofs(node):
                fixed.add(dof)
                apply_dirichlet_bc(K, f, dof, 0.0)

    elif base_bc == "robin":
        def on_base(p, q):
            return abs(p[1]) < TOL and abs(q[1]) < TOL

        apply_robin_bc(K, elements, on_base, KN, KT)

    else:
        raise ValueError("base_bc desconocida")

    # --- Resolver ---
    u = solve_sparse(K, f)

    # --- Reacciones ---
    R = K0 @ u - f0
    Rx = sum(R[2 * node] for node in base_nodes)
    Ry = sum(R[2 * node + 1] for node in base_nodes)

    # --- Momentos analíticos ---
    area = 0.5 * (b + c) * H
    W = rho_h * G * area * THICKNESS
    x_cg = (b**2 + b*c + c**2) / (3 * (b + c))

    M_est = W * (b - x_cg)

    Fempuje = 0.5 * rho_w * G * Hagua**2 * THICKNESS
    Hempuje = Hagua / 3
    M_des = Fempuje * Hempuje

    FS = M_est / M_des

    # Momento FEM
    M_react = sum((mesh.node(node).x - b) * R[2 * node + 1] for node in base_nodes)

    # --- Tensiones ---
    sigma1 = []
    sigma2 = []
    sigma_vm = []

    for el in elements:
        ps = el.principal_stress(u)   # ps = [s1, s2, szz]
        sigma1.append(ps[0])
        sigma2.append(ps[1])
        sigma_vm.append(el.vonMises_stress(u))

    sigma1_max = max(sigma1)
    sigma2_min = min(sigma2)
    vm_max = max(sigma_vm)

    idx = int(numpy.argmax(sigma1))
    xc, yc = elements[idx].X.mean(axis=0)

    # --- Desplazamiento máximo ---
    umax = numpy.abs(u).max()

    # --- Devolver todo ---
    return dict(
        element=element,
        formulation=formulation,
        base_bc=base_bc,
        nx=nx,
        ny=ny,
        ny_top=ny_top,
        n_nodes=n_nodes,
        n_elements=len(elements),

        u=u,
        R=R,
        Rx=Rx,
        Ry=Ry,

        FS=FS,
        M_est=M_est,
        M_des=M_des,
        M_react=M_react,

        sigma1_max=sigma1_max,
        sigma2_min=sigma2_min,
        vm_max=vm_max,
        sigma1_list=sigma1,
        sigma2_list=sigma2,
        vm_list=sigma_vm,
        crit_x=xc,
        crit_y=yc,

        umax=umax)
