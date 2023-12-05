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
            visited = {initial.get_name(): 0}
            path = {}

            nodes = set(self.graph_dict.keys())

            while nodes:
                min_node = None
                for node in nodes:
                    if node.get_name() in visited:
                        if min_node is None:
                            min_node = node
                        elif visited[node.get_name()] < visited[min_node.get_name()]:
                            min_node = node

                if min_node is None:
                    break

                nodes.remove(min_node)
                current_weight = visited[min_node.get_name()]

                for neighbor, weight in self.graph_dict[min_node]:
                    total_weight = current_weight + weight
                    if neighbor.get_name() not in visited or total_weight < visited[neighbor.get_name()]:
                        visited[neighbor.get_name()] = total_weight
                        path[neighbor.get_name()] = min_node.get_name()

            return visited, path

    def bellman_ford(self, source):
        distance = {vertex.get_name(): float('inf') for vertex in self.graph_dict}
        distance[source.get_name()] = 0

        for _ in range(len(self.graph_dict) - 1):
            for vertex in self.graph_dict:
                for neighbour, weight in self.graph_dict[vertex]:
                    if distance[vertex.get_name()] != float('inf') and distance[vertex.get_name()] + weight < distance[neighbour.get_name()]:
                        distance[neighbour.get_name()] = distance[vertex.get_name()] + weight

        for vertex in self.graph_dict:
            for neighbour, weight in self.graph_dict[vertex]:
                if distance[vertex.get_name()] != float('inf') and distance[vertex.get_name()] + weight < distance[neighbour.get_name()]:
                    print("El grafo contiene un ciclo de peso negativo")
                    return

        return distance

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
    
    def kruskal(self):
        edges = []#almacena aristas
        for vertex, neighbors in self.graph_dict.items():#recorre el grafo
            for neighbor, weight in neighbors:
                edges.append((weight, vertex, neighbor))#añade como tupla el peso y los vertices que forman la arista

        edges.sort(key=lambda edge: edge[0])#ordena las aristas en orden ascendente

        parent = {vertex: vertex for vertex in self.graph_dict}

        def find(v):# encuentra la raíz del conjunto al que pertenece un vértice
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]

        def union(v1, v2):#une dos conjuntos disjuntos especificando sus representantes.
            root1 = find(v1)
            root2 = find(v2)
            parent[root1] = root2

        minimum_spanning_tree = []#arbol de expansion minima
        total_weight = 0#peso total

        for edge in edges:#
            weight, vertex_1, vertex_2 = edge
            if find(vertex_1) != find(vertex_2):#VERIFICA si son del mismo conjunto disconjunto(estudiar)
                union(vertex_1, vertex_2)#si no es asi los une y agrega al arbol de expansion minima
                minimum_spanning_tree.append((vertex_1.get_name(), vertex_2.get_name(), weight))
                total_weight += weight#y suma el peso

        return minimum_spanning_tree, total_weight


# -----------------------Programa principal--------------------------------
g = Graph(directed=False)

# # Creando vértices
# vertex_a = Vertex('a')
# vertex_b = Vertex('b')
# vertex_c = Vertex('c')
# vertex_d = Vertex('d')

# # Añadir vértices al grafo                       
# g.add_vertex(vertex_a)
# g.add_vertex(vertex_b)
# g.add_vertex(vertex_c)
# g.add_vertex(vertex_d)

# # Creamos aristas a partir de los vértices existentes en el grafo 
# g.add_edge(Edge(vertex_a, vertex_b, 1))
# g.add_edge(Edge(vertex_a, vertex_d, 5))
# g.add_edge(Edge(vertex_b, vertex_c, 4))
# g.add_edge(Edge(vertex_b, vertex_d, 5))
# g.add_edge(Edge(vertex_c, vertex_d, 2))
# g.add_edge(Edge(vertex_b, vertex_d, 5))

# #ejemplo uso de kruskal
# v1 = Vertex("A")
# v2 = Vertex("B")
# v3 = Vertex("C")
# v4 = Vertex("D")

# g.add_vertex(v1)
# g.add_vertex(v2)
# g.add_vertex(v3)
# g.add_vertex(v4)

# edge1 = Edge(v1, v2, 10)
# edge2 = Edge(v1, v3, 6)
# edge3 = Edge(v1, v4, 5)
# edge4 = Edge(v2, v4, 15)
# edge5 = Edge(v3, v4, 4)

# g.add_edge(edge1)
# g.add_edge(edge2)
# g.add_edge(edge3)
# g.add_edge(edge4)
##prueba de kruskal y prim si dan pesos distintos dado un grafo especifico
vertex_a = Vertex('A')
vertex_b = Vertex('B')
vertex_c = Vertex('C')
vertex_d = Vertex('D')

# Añadir vértices al grafo                       
g.add_vertex(vertex_a)
g.add_vertex(vertex_b)
g.add_vertex(vertex_c)
g.add_vertex(vertex_d)

# Creamos aristas a partir de los vértices existentes en el grafo 
g.add_edge(Edge(vertex_a, vertex_b, 4))
g.add_edge(Edge(vertex_b, vertex_c, 2))
g.add_edge(Edge(vertex_b, vertex_d, 1))
g.add_edge(Edge(vertex_b, vertex_a, 5))
g.add_edge(Edge(vertex_c, vertex_d, 1))


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
minimum_spanning_tree, total_weight = g.kruskal()
print("arbol de expansion minima(Kruskal): ",minimum_spanning_tree)
print("Peso total(Kruskal):",total_weight)

# Dijkstra
#print(g.dijkstra(vertex_a))
# Bellman-Ford
#print(g.bellman_ford(vertex_a))