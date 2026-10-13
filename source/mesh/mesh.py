from source.mesh.node import Node
from source.materials.linear_elastic import LinearElastic

# (familia, orden de interpolacion) -> tipo de elemento
ELEMENT_TYPES = {("TRI", 1): "TRI3", ("TRI", 2): "TRI6", ("QUAD", 1): "QUAD4", ("QUAD", 2): "QUAD8"}

# tipo de material -> clase
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
        self.analysis = "plane_stress"
        self.thickness = 1.0

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
        dim = int(header[2])
        n_materials= 0
        if len(header)>3:
            n_materials = int(header[3])

        self.dim = dim

        line_number = 1
        if n_materials> 0: 
            values= lines[line_number].split()
            line_number=line_number + 1
            self.analysis = values[0]
            self.thickness = float(values[1])
            for i in range(n_materials):
                values=lines[line_number].split()
                line_number= line_number +1
                material_id= int(values[0])
                material_type=values[1]
                E =float(values[2])
                nu= float(values[3])
                rho= float(values[4])
                self.materials[material_id]= MATERIAL_TYPES[material_type](E, nu, rho)

        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            global_id = int(values[0]) - 1
            if global_id != i:
                raise ValueError("Unexpected node numbering")

            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

        for i in range(n_elements):
            values = lines[line_number].split()
            line_number = line_number + 1

            element_global_id = int(values[0]) - 1
            if n_materials >0: 
                family= values[1]
                order = int(values[2])
                material_id = int(values[3])
                first_node=4
                if (family, order) not in ELEMENT_TYPES:
                    raise ValueError("Unknown element: %s of order %d" % (family, order))
                element_type = ELEMENT_TYPES[(family, order)]
            else:
                element_type = values[1]
                material_id= None
                first_node = 2
            element_conn=[]
            for j in range(first_node, len(values)):
                element_conn.append(int(values[j]) - 1)

            self.element_global_ids.append(element_global_id)
            self.element_types.append(element_type)
            self.element_materials.append(material_id)
            self.connectivity.append(element_conn)

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
