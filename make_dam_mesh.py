"""
Generador de mallas estructuradas para la presa de gravedad
"""
import sys

#---Geometría por defecto-----
HEIGHT=30.0
BASE=18.0
CREST=3.0
FREEBOARD=2.0
UPSTREAM_BATTER=0.0
#--- Material por defecto: hormigón H21---
MATERIAL=("linear_elastic", 21.0e9, 0.25, 2400.0)
ANALYSIS="plane_strain"
THICKNESS=1.0

ELEMENT_NAMES={"T3":"TRI3", "T6":"TRI6", "Q4":"QUAD4", "Q8":"QUAD8"}
FAMILY_ORDER={"TRI3":("TRI",1), "TRI6":("TRI",2), "QUAD4":("QUAD",1), "QUAD8":("QUAD",2)}


def dam_mesh(element="Q4", nx=4, ny=8, ny_top=1, height=HEIGHT, base=BASE, crest=CREST, freeboard=FREEBOARD, upstream_batter=UPSTREAM_BATTER):
  if element not in ELEMENT_NAMES:
    raise ValueError("Elemento desconocido: %s (usar T3, T6, Q4 o Q8)" %element)
  water=height-freeboard
  quadratic=element in ("T6", "Q8")

  def x_left(y):
    return upstream_batter*(1.0-y/height)
  def x_right(y):
    return base+(crest-base)*y/height

  def y_of(J):
    if J<=2*ny:
      return water*J/(2*ny)
    return water + freeboard*(J-2*ny)/(2*ny_top)

  def coords(I, J):
    y=y_of(J)
    s=I/(2*nx)
    return x_left(y) + (x_right(y)-x_left(y))*s,y

  numbers={}
  nodes=[]

  def node(I,J): 
    if (I,J) not in numbers:
      nodes.append(coords(I, J))
      numbers[(I,J)] = len(nodes)
    return numbers[(I,J)]
  elements=[]
  for j in range (0, 2*(ny+ny_top), 2):
    for i in range(0, 2*nx, 2):
      a, b, c, d= (i,j), (i +2, j), (i +2, j+2), (i, j+2)
      ab, bc, cd, da= (i+1, j), (i+2, j+1), (i+1, j+2), (i, j+1)
      center=(i+1, j+1)

      if element == "Q4": 
        elements.append([node(*p) for p in (a, b, c, d)])
      elif element == "Q8":
        elements.append([node(*p) for p in (a, b, c, d, ab, bc, cd, da)])
      elif element == "T3":
        elements.append([node(*p) for p in (a, b, c)])
        elements.append([node(*p) for p in (a, c, d)])
      else: 
        elements.append([node(*p) for p in (a, b, c, ab, bc, center)])
        elements.append([node(*p) for p in (a, c, d, center, cd, da)])

  return nodes, elements, ELEMENT_NAMES[element]

def write_mesh(filename, nodes, elements, element_name, header=""): 
  family, order = FAMILY_ORDER[element_name]
  with open(filename, "w") as f: 
    if header:
      for line in header.strip().splitlines():
        f.write("# %s\n" % line)
      f.write("\n")
    f.write("# n_nodes n_elements dim n_materials\n")
    f.write("%d %d 2 1\n\n" % (len(nodes), len(elements)))
    f.write("# analysis thickness\n")
    f.write("%s %g\n\n" % (ANALYSIS, THICKNESS))
    f.write("# material_id type E nu rho\n")
    f.write("1 %s %g %g %g\n\n" % MATERIAL)
    f.write("# node_number x y\n")
    for k, (x, y) in enumerate(nodes,1):
      f.write("%d %.6f %.6f\n" % (k, x, y))
    f.write("\n# element_number family order material_id connectivity\n")
    for k, conn in enumerate(elements, 1):
      f.write("%d %-4s %d 1 %s\n" % (k, family, order, " ".join(str(c) for c in conn)))
      

def make_dam_mesh (element="Q4", nx=4, ny=8, ny_top=1, filename=None, **geometry): 
  nodes, elements, name = dam_mesh(element, nx, ny, ny_top, **geometry)
  if filename is None:
    filename= "meshes/dam_%s.mesh" % element

  g=dict(height=HEIGHT, base=BASE, crest=CREST, freeboard=FREEBOARD, upstream_batter=UPSTREAM_BATTER)
  g.update(geometry)
  header=("Presa de gravedad generada con make_dam_mesh.py\n"
          "altura %g, base %g, cresta %g, borde libre %g, talud aguas arriba %g\n"
          "%s, nx=%d, ny=%d (bajo el agua) + %d (borde libre)\n"
          "Cara aguas arriba en x= %g (en la base), base en y=0."
          %(g["height"], g["base"], g["crest"], g["freeboard"], g["upstream_batter"], name, nx, ny, ny_top, g["upstream_batter"]))
  write_mesh(filename, nodes, elements, name, header)
  print("%s: %d nodos, %d elementos -> %s" %(name, len(nodes), len(elements), filename))
  return filename


if __name__ == "__main__":
  if len(sys.argv)>1:
    element=sys.argv[1]
    nx=int(sys.argv[2]) if len(sys.argv)> 2 else 4
    ny= int(sys.argv[3]) if len(sys.argv)> 3 else 8
    make_dam_mesh(element, nx, ny)
  else:
    for element in ("T3", "T6", "Q4", "Q8"):
      make_dam_mesh(element, nx=4, ny=8)
