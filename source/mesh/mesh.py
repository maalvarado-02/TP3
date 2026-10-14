from source.mesh.node import Node
from source.materials.linear_elastic import LinearElastic

# (tipo elemento, orden de interpolacion) -> tipo de elemento
ELEMENT_TYPES = {("TRI", 1): "TRI3", ("TRI", 2): "TRI6", ("QUAD", 1): "QUAD4", ("QUAD", 2): "QUAD8"}

# Tipo de material -> clase
MATERIAL_TYPES = {"linear_elastic": LinearElastic}


class Mesh:
    def __init__(self):
        self.dim = 0
        self.dofs_per_node = 0
        self.nodes = []
        self.element_global_ids = []
        self.element_types = []
        self.connectivity = []
        self.element_materials = []
        self.materials = {}

    def read(self, filename):
        file = open(filename, "r")

        lines = []

        for raw_line in file:
            line = raw_line.split("#")[0]
            line = line.strip()

            if line != "":
                lines.append(line)

        file.close()

        header = lines[0].split()

        n_nodes = int(header[0])
        n_elements = int(header[1])
        self.dim = int(header[2])
        n_materials = int(header[3])

        if self.dim != 2:
            raise ValueError("Solo se soporta 2D")
        
        line_number = 1

        # --- Materiales ---
        for i in range(n_materials):
            values = lines[line_number].split()
            line_number += 1

            material_id = int(values[0])
            material_type = values[1]
            if material_type not in MATERIAL_TYPES:   # Chequeo: tipo de material válido
                raise ValueError(f"Tipo de material desconocido: {material_type}")

            E = float(values[2])
            nu = float(values[3])
            rho = float(values[4])
            self.materials[material_id] = MATERIAL_TYPES[material_type](E, nu, rho)

        # --- Nodos ---
        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            global_id = int(values[0]) - 1
            if global_id != i:
                raise ValueError("Unexpected node numbering")

            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

        # --- Elementos ---
        for i in range(n_elements):
            values = lines[line_number].split()
            line_number += 1

            element_global_id = int(values[0]) - 1
            family = values[1]
            order = int(values[2])
            material_id = int(values[3])

            if material_id not in self.materials:   # Chequeo: material asignado existe
                raise ValueError(f"Material ID {material_id} no definido")

            if (family, order) not in ELEMENT_TYPES:   # Chequeo: tipo de elemento válido
                raise ValueError("Tipo de elemento desconocido")
            
            element_type = ELEMENT_TYPES[(family, order)]
            
            element_conn = []
            for j in range(4, len(values)):
                element_conn.append(int(values[j]) - 1)
            
            self.element_global_ids.append(element_global_id)
            self.element_types.append(element_type)
            self.element_materials.append(material_id)
            self.element_materials.append(material_id)
            self.connectivity.append(element_conn)

    # Funciones auxiliares para acceder a la información de la malla
    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_elements(self):
        return len(self.connectivity)

    def node(self, node_global_id):
        return self.nodes[node_global_id]

    def element_material(self, e):
        return self.materials[self.element_materials[e]]

    def number_of_dofs(self):
        return self.number_of_nodes() * self.dofs_per_node

    def node_global_dofs(self, node_global_id):
        dofs = []

        for i in range(self.dofs_per_node):
            dofs.append(node_global_id * self.dofs_per_node + i)

        return dofs
