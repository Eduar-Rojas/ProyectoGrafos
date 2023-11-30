import heapq 
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
    def __init__(self, vertex_1, vertex_2, weight):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.weight = weight

    # Obtener datos de la clase
    def __str__(self):
        return self.vertex_1.get_name() + " ---> " + self.vertex_2.get_name()

    # Gets and sets
    def get_vertex_1(self):
        return self.vertex_1
    
    def get_vertex_2(self):
        return self.vertex_2

    def get_weight(self):
        return self.weight

class Graph:
    # Constructor, crea un grafo con un diccionario
    def __init__(self, directed=True):
        self.graph_dict = {}
        self.directed = directed

    # Obtener datos del grafo
    def __str__(self):
        graph_dict = ""
        for vertex, neighbors in self.graph_dict.items():
            for neighbor, weight in neighbors:
                graph_dict += f"{vertex.get_name()} ---({weight})---> {neighbor.get_name()}\n"
        return graph_dict

    # Añade un vértice al grafo (él vertice debe estár creado)
    def add_vertex(self, vertex):
        if vertex in self.graph_dict:
            return print("El Vertice ya está en el grafo:",vertex)
        self.graph_dict[vertex] = []
        print(f"el vertice: {vertex} se agregó al diccionario")

    # Añade una arista al grafo (la arista debe estár creada)
    def add_edge(self, edge):
        vertex_1 = edge.get_vertex_1()
        vertex_2 = edge.get_vertex_2()
        weight = edge.get_weight()
        self.graph_dict[vertex_1].append((vertex_2, weight))
        if not self.directed:
            self.graph_dict[vertex_2].append((vertex_1, weight))  # Agregar esta línea si el grafo es no dirigido

    # verificar si un vertice existe
    def get_vertex(self, vertex_name):
        for v in self.graph_dict:
            if vertex_name == v.get_name():
                return print("existe ", v)
        print(f"El vétice {vertex_name} no existe")

    #Topological sort
    

    def BFS(self, start):
        visited = []
        queue = []

        visited.append(start)
        queue.append(start)

        while queue:
            s = queue.pop(0)
            print(s, end=" ")

            for neighbour in self.graph_dict[s]:
                neighbour = neighbour[0]
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
                neighbour = neighbour[0]
                if neighbour not in visited:
                    self.DFS(neighbour, visited)

    def dijkstra(self, initial):
        print("Algoritmo Dijkstra ejecutándose: ")
        visited = {initial.get_name(): 0}
        path = {}

        nodes = set(self.graph_dict.keys())

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None or visited[node] < visited[min_node]:
                        min_node = node

            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for neighbor, weight in self.graph_dict[min_node]:
                total_weight = current_weight + weight
                if neighbor not in visited or total_weight < visited[neighbor]:
                    visited[neighbor] = total_weight
                    path[neighbor] = min_node

        return visited, path

    def prim(self):
        visitados = set()
        arbolSM = []
        suma_total = 0
        nodoI = list(self.graph_dict.keys())[0]
        cola = [(0, nodoI)]

        while cola:
            peso, nodoA = heapq.heappop(cola)

            if nodoA not in visitados:
                visitados.add(nodoA)
                arbolSM.append((nodoA.get_name(), peso))  # Cambiar aquí
                suma_total += peso

                if nodoA in self.graph_dict:
                    for vecino, pesoB in self.graph_dict[nodoA]:
                        if vecino not in visitados:
                            heapq.heappush(cola, (pesoB, vecino))

        return arbolSM, suma_total

# -----------------------Programa principal--------------------------------
g = Graph(directed=False)

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
g.add_edge(Edge(vertex_a, vertex_b, 1))
g.add_edge(Edge(vertex_a, vertex_d, 5))
g.add_edge(Edge(vertex_b, vertex_c, 4))
g.add_edge(Edge(vertex_b, vertex_d, 5))
g.add_edge(Edge(vertex_c, vertex_d, 2))
g.add_edge(Edge(vertex_b, vertex_d, 5))

# Si fuera no dirigido:
# graph_dict = {'a': [('b', 1), ('c', 2)],
#               'b': [('a', 1), ('d', 3)],
#               'c': [('a', 2)]}

# Si fuera dirigido:
# graph_dict = {'a': [('b', 1), ('c', 2)],
#               'b': [('d', 3)],
#               'c': [()]}

# Método del grafo BFS

#g.BFS(vertex_a)

# Método del grafo DFS

#g.DFS(vertex_c)

# Mostrar adyacencias de vértices

print(g)

arbol_expansion_minima, suma_total = g.prim()
print("Árbol de expansión mínima:", arbol_expansion_minima)
print("Suma total del árbol:", suma_total)

print(g.dijkstra(vertex_a))

# ({'A': 0, 'B': 1, 'C': 2, 'D': 4}, {'B': 'A', 'C': 'A', 'D': 'B'})
# ({'a': 0}, {})