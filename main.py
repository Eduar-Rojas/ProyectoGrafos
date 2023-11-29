class Vertex:
    # Constructor
    def __init__(self, name):
        self.name = name
    # Obtener datos de la clase
    def __str__(self):
        return self.name
    #Gets and sets
    def get_name(self):
        return self.name

class Edge:
    #Constructor
    def __init__(self, vertex_1, vertex_2):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2

    # Obtener datos de la clase
    def __str__(self):
        return self.vertex_1.get_name() + " ---> " + self.vertex_2.get_name()

    #Gets and sets
    def get_vertex_1(self):
        return self.vertex_1
    def get_vertex_2(self):
        return self.vertex_2