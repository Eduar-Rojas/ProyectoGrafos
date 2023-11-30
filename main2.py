class Vertex:
    # Constructor
    def __init__(self, name):
        self.name = name

    # Obtener datos de la clase
    def __str__(self):
        return self.name
    
    # Gets and sets
    def get_name(self):
        return self.name

class Edge:
    # Constructor
    def __init__(self, vertex_1, vertex_2):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2

    # Obtener datos de la clase
    def __str__(self):
        return self.vertex_1.get_name() + " ---> " + self.vertex_2.get_name()

    # Gets and sets
    def get_vertex_1(self):
        return self.vertex_1
    
    def get_vertex_2(self):
        return self.vertex_2

class Graph:
    # Constructor, crea un grafo con un diccionario
    def __init__(self):
        self.graph_dict = {}

    def __str__(self):
        all_edges = ""
        for vertex in self.graph_dict:
            for neighbors in self.graph_dict[vertex]:
                all_edges += vertex.get_name() + " ---> " + neighbors.get_name() + "\n"
        return all_edges

    # Añade un vértice al grafo
    def add_vertex(self, vertex):
        if vertex in self.graph_dict:
            return "El vértice ya está en el grafo"
        self.graph_dict[vertex] = []

    def add_edge(self, edge):
        v1 = edge.get_vertex_1()
        v2 = edge.get_vertex_2()
        self.graph_dict[v1].append(v2)
        self.graph_dict[v2].append(v1)  # Agregar esta línea si el grafo es no dirigido

    def get_vertex(self, vertex_name):
        for v in self.graph_dict:
            if vertex_name == v.get_name():
                return v
        print(f"El vétice {vertex_name} no existe")

    def BFS(self, start):
        visited = []
        queue = []

        visited.append(start)
        queue.append(start)

        while queue:
            s = queue.pop(0)
            print(s, end=" ")

            for neighbour in self.graph_dict[s]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)

    def DFS(self, start, visited=None):
        if visited is None:
            visited = set()

        if start not in visited:
            print(start)
            visited.add(start)

            for neighbour in self.graph_dict[start]:
                if neighbour not in visited:
                    self.DFS(neighbour, visited)

# Programa principal
g = Graph()

# Creando vértices
vertex_a = Vertex('a')
vertex_b = Vertex('b')
vertex_c = Vertex('c')
vertex_d = Vertex('d')

# Añadir vértices al grafo 
g.add_vertex(vertex_a)
g.add_vertex(vertex_b)
g.add_vertex(vertex_c)
g.add_vertex(vertex_d)

# Creamos aristas a partir de los vértices existentes en el grafo
g.add_edge(Edge(vertex_a, vertex_b))
g.add_edge(Edge(vertex_a, vertex_c))
g.add_edge(Edge(vertex_b, vertex_d))

# Método del grafo BFS

#g.BFS(vertex_a)

# Método del grafo DFS

#g.DFS(vertex_a)

# Mostrar adyacencias de vértices

print(g)