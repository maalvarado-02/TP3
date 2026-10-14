import sys

# Generador de mallas estructuradas para la presa de gravedad

#---Geometría por defecto-----
HEIGHT = 30
BASE = 18
CREST = 3
FREEBOARD = 2
UPSTREAM_BATTER = 0

#--- Material por defecto: hormigón H21---
MATERIAL = ("linear_elastic", 21.0e9, 0.2, 2400)

ELEMENT_NAMES = {"T3":"TRI3", "T6":"TRI6", "Q4":"QUAD4", "Q8":"QUAD8"}

def dam_mesh(element, nx, ny, ny_top, height, base, crest, freeboard, upstream_batter):
  if element not in ELEMENT_NAMES:
    raise ValueError("Elemento desconocido: %s (usar T3, T6, Q4 o Q8)" %element)
  
  water = height - freeboard

  def x_left(y):
    return upstream_batter * (1 - y/height)
  def x_right(y):
    return base + (crest-base) * y/height
  def y_of(J):
    if J <= 2*ny:
      return water * J/(2*ny)
    return water + freeboard*(J-2*ny)/(2*ny_top)

  def coords(I, J):
    y=y_of(J)
    xl = x_left(y)
    xr = x_right(y)
    s = I / (2*nx)
    return xl + (xr - xl)*s, y

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
  with open(filename, "w") as f: 
    
    if header:
      for line in header.strip().splitlines():
        f.write("# %s\n" % line)
      f.write("\n")

    f.write("# n_nodes | n_elements | dim | n_materials\n")
    f.write("%d %d 2 1\n\n" % (len(nodes), len(elements)))

    f.write("# material_id | type | E | nu | rho\n")
    f.write("1 %s %g %g %g\n\n" % MATERIAL)

    f.write("# node_number | x | y\n")
    for k, (x, y) in enumerate(nodes,1):
      f.write("%d %.6f %.6f\n" % (k, x, y))
    
    f.write("\n# element_number | element_type | material_id | connectivity\n")
    for k, conn in enumerate(elements, 1):
      f.write("%d %-5s 1 %s\n" % (k, element_name , " ".join(str(c) for c in conn)))
      

def make_dam_mesh (element="Q4", nx=4, ny=8, ny_top=1, filename=None, height=HEIGHT, base=BASE, crest=CREST,
                  freeboard=FREEBOARD, upstream_batter=UPSTREAM_BATTER): 
  nodes, elements, element_name = dam_mesh(element, nx, ny, ny_top, height, base, crest, freeboard, upstream_batter)
  
  if filename is None:
    filename= "meshes/dam_%s.mesh" % element
  
  header = (
        "Presa de gravedad generada con make_dam_mesh.py\n"
        f"Altura(H): {height}, Base(b): {base}, Cresta(c): {crest}, "
        f"Borde libre(d): {freeboard}, Talud aguas arriba: {upstream_batter}\n"
        f"{element_name}, nx = {nx}, ny = {ny} (bajo el agua) + {ny_top} (borde libre)\n"
        f"Cara aguas arriba en x = {upstream_batter} (en la base), base en y = 0."
    )
  write_mesh(filename, nodes, elements, element_name, header)
  print("%s: %d nodos, %d elementos -> %s" %(element_name, len(nodes), len(elements), filename))
  return filename

# Crea todas las mallas en simultaneo (en vez de llamarlo desde dam.py)
"""
if __name__ == "__main__":
  if len(sys.argv)>1:
    element=sys.argv[1]
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ny = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    make_dam_mesh(element, nx, ny, ny_top=1, filename=None, height=HEIGHT, base=BASE, crest=CREST,
                  freeboard=FREEBOARD, upstream_batter=UPSTREAM_BATTER)
  else:
    for element in ("T3", "T6", "Q4", "Q8"):
      make_dam_mesh (element, nx=4, ny=8, ny_top=1, filename=None, height=HEIGHT, base=BASE, crest=CREST,
                  freeboard=FREEBOARD, upstream_batter=UPSTREAM_BATTER)
"""