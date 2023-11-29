class Vertex:
    def __init__(self, name):
        self.name = name
    def get_name(self):
        return self.name
    def __str__(self):
        return self.name

class Edge:
    def __init__(self, vertex_1, vertex_2):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2    
    def get_vertex_1(self):
        return self.vertex_1
    def get_vertex_2(self):
        return self.vertex_2
    def __str__(self):
        return self.vertex_1.get_name() + " ---> " + self.vertex_2.get_name()