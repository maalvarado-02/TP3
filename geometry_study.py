import numpy
from scipy.sparse.linalg import spsolve
 
from source.mesh.mesh import Mesh
from source.elements.solid2d import Solid2D
from source.systems.matrix import SparseMatrix
from source.systems.loads import assemble_body_force, assemble_pressure
from make_dam_mesh import make_dam_mesh

#estudio de variaciones de geometría de la presa. Para cada geometría, va a generar una malla, la resuelve con el modelo FEM y se compara el factor de seguridad
#al vuelco analítico con el equilibrio del FEM

#---Datos fijos-----
ELEMENT="Q8"
FORMULATION ="plane_strain"
nx=8
ny=16
ny_top=1

THICKNESS=1.0
G=9.81
rhoAgua=1000
rhoHormigon=2400
H=30
d=2
Hagua=H-d
TOL=1.0e-6

#--- Geometrías a estudiar 
CASES=[
  ("Original", 18, 3, 0),
  ("Base 20 m", 20, 3, 0),
  ("Base 22 m", 22, 3, 0),
  ("Base 24 m", 24, 3, 0),
  ("Cresta 5 m", 18, 5, 0),
  ("Cresta 7 m", 18, 7, 0),
  ("Talud 3 m", 18, 3, -3),
  ("Talud 6 m", 18, 3, -6),
]

def polygon_area_centroid(points):
  A=0.0
  Sx=0.0
  for i in range(len(points)):
    x1, y1 = points[i]
    x2, y2 = points[(i+1)%len(points)]
    cross=x1*y2 - x2*y1
    A += cross
    Sx += (x1+x2)*cross
  A=A/2
  return A, Sx/(6*A)


def analytic(b, c, t):
  area, x_g= polygon_area_centroid([(t,0),(b,0),(c,H),(0,H)])
  W=rhoHormigon*G*area*THICKNESS

  x_w=t*(1-Hagua/H)
  area_w=0.5*Hagua*(x_w-t)
  W_w= rhoAgua*G*area_w*THICKNESS
  x_gw=(2*t+x_w)/3

  M_est=W*(b-x_g)+W_w*(b-x_gw)
  
  #empuje del agua
  F=0.5*rhoAgua*G*Hagua**2*THICKNESS
  M_des=F*Hagua/3

  return area, W + W_w, M_est, M_des


def fem(b,c,t):
  mesh_file=make_dam_mesh(ELEMENT, nx, ny, ny_top, "meshes/dam_study.mesh", H, b, c, d, t)
  mesh=Mesh()
  mesh.read(mesh_file)
  mesh.dofs_per_node=Solid2D.dofs_per_node
  n_dofs=mesh.number_of_dofs()

  elements=[]
  for e in range(mesh.number_of_elements()):
    elements.append(Solid2D(mesh.element_global_ids[e], mesh.connectivity[e], mesh, mesh.element_material(e), THICKNESS, mesh.element_types[e], FORMULATION))

  K=SparseMatrix(n_dofs)
  for el in elements:
    K.scatter_local_to_global(el)
  K=K.to_csc()


  #cargas: peso propio y presión hidrostática
  f=numpy.zeros(n_dofs)
  assemble_body_force(f, elements, (0.0, -rhoHormigon*G))

  def x_face(y):
    return t*(1-y/H)
  def on_upstream_face(p,q):
    return (abs(p[0] - x_face(p[1])) < TOL and abs(q[0] - x_face(q[1])) < TOL and p[1] <= Hagua + TOL and q[1] <= Hagua + TOL)

  def hydrostatic_pressure(x,y):
    return rhoAgua*G*(Hagua-y)

  assemble_pressure(f, elements, on_upstream_face, hydrostatic_pressure)

  base_nodes=[i for i in range(mesh.number_of_nodes() ) if abs(mesh.node(i).y)<TOL]
  fixed = set()
  for node in base_nodes:
    for dof in mesh.node_global_dofs(node):
      fixed.add(dof)
  free = [i for i in range(n_dofs) if i not in fixed]

  u = numpy.zeros(n_dofs)
  u[free]=spsolve(K[free, :][:,free], f[free])

  R= K @ u - f
  Ry= sum(R[2*i+1] for i in base_nodes)
  M_reac=sum((b-mesh.node(i).x)*R[2*i +1] for i in base_nodes)

  sigma1=max(el.principal_stress(u)[0] for el in elements)
  return Ry, M_reac, sigma1, numpy.abs(u).max()


#--- Estudio de variacion ----
rows=[]
area0=None
for name, b, c, t in CASES:
  area, V, M_est, M_des = analytic(b, c, t)
  Ry, M_reac, sigma1, umax= fem(b, c, t)

  if area0 is None:
    area0=area

  FS= M_est/M_des
  L=b-t
  e=M_reac/Ry
  rows.append((name, area, 100*(area/area0-1), FS, M_est-M_des, M_reac, e, L, sigma1, umax))

print("\n%-12s %8s %8s %7s %12s %12s %8s %10s %10s %8s" % ("Geometría", "Area", "dArea%", "FS", "Mneto an.", "Mreac FEM", "e [m]", "L/3-2L/3", "s1max", "umax"))
print("%-12s %8s %8s %7s %12s %12s %8s %10s %10s %8s" % ("", "[m2]", "", "", "[MN.m]", "[MN.m]", "", "[m]", "[MPa]", "[mm]"))

for r in rows:
  name, area, darea, FS, M_neto, M_reac, e, L, sigma1, umax=r
  print("%-12s %8.1f %8.1f %7.3f %12.3f %12.3f %8.2f %4.1f-%-5.1f %10.3f %8.3f" % (name, area, darea, FS, M_neto / 1e6, M_reac / 1e6, e, L / 3, 2 * L / 3,sigma1 / 1e6, 1e3 * umax))

#--- Tabla con resultados ----
import matplotlib.pyplot as plt
headers=["Geometría", "Area\[m^2]", "ΔÁrea\n[%]", "FS", "M neto\n[MN.m]", "M reac FEM\n[MN.m]", "e\n[m]", "Tercio central\n[m]"]
cells=[]
for r in rows: 
 name, area, darea, FS, M_neto, M_reac, e, L, sigma1, umax =r
 cells.append([name, "%.1f" % area, "%.1f" % darea, "%.2f" % FS, "%.2f" % (M_neto/1e6), "%.2f" % (M_reac/1e6), "%.2f" % e, "%.1f - %.1f" % (L/3, 2*L/3)])

fig, ax= plt.subplots(figsize=(11, 0.*len(rows) +1.5))
ax.axis("off")
table= ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(len(headers)):
 table[0,j].set_facecolor("#d9d9d9")
 table[0,j].set_text_props(weight="bold")

ax.set_title("Factor de seguridad al vuelco para distintas geometrías", fontweight="bold")
plt.tight_layout()
plt.show()
