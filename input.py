import heapq
# Clase VERTICE
class Vertex:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def get_name(self):
        return self.name

# Clase ARISTA
class Edge:
    def __init__(self, vertex_1, vertex_2, weight):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.weight = weight

    def __str__(self):
        return f"{self.vertex_1.get_name()} ---({self.weight})---> {self.vertex_2.get_name()}"

    def get_vertex_1(self):
        return self.vertex_1

    def get_vertex_2(self):
        return self.vertex_2

    def get_weight(self):
        return self.weight

# Clase GRAFO
class Graph:
    def __init__(self, directed=True):
        self.graph_dict = {}
        self.directed = directed

    def __str__(self):
        graph_str = ""
        for vertex, neighbors in self.graph_dict.items():
            for neighbor, weight in neighbors:
                graph_str += f"{vertex.get_name()} ---({weight})---> {neighbor.get_name()}\n"
        return graph_str

    def add_vertex(self, vertex):
        if vertex in self.graph_dict:
            return print("El vértice ya está en el grafo:", vertex)
        self.graph_dict[vertex] = []
        print(f"El vértice {vertex} se agregó al grafo.")

    def add_edge(self, edge):
        vertex_1 = edge.get_vertex_1()
        vertex_2 = edge.get_vertex_2()
        weight = edge.get_weight()
        self.graph_dict[vertex_1].append((vertex_2, weight))
        if not self.directed:
            self.graph_dict[vertex_2].append((vertex_1, weight))

    def get_vertex(self, vertex_name):
        for v in self.graph_dict:
            if vertex_name == v.get_name():
                return print("Existe ", v)
        print(f"El vértice {vertex_name} no existe")

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

    def DFS(self, start, stack=None, visited=None, order=None):
        if visited is None:
            visited = set()
        if stack is None:
            stack = []
        if order is None:
            order = []

        if start not in visited:
            order.append(start.get_name())
            visited.add(start)

            for neighbour in self.graph_dict[start]:
                neighbour = neighbour[0]
                if neighbour not in visited:
                    self.DFS(neighbour, stack, visited, order)

            stack.append(start.get_name())
        return order

    def topological_sort(self):
        if self.directed:
            stack = []
            visited = set()

            for vertex in self.graph_dict:
                if vertex not in visited:
                    self.DFS(vertex, stack, visited)

            ordering = []
            while stack:
                ordering.append(stack.pop())
            return ordering
        else:
            return "El grafo no es DAG"

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

    def prim(self, initial):
        visited = set()
        minimum_spanning_tree = []
        total_weight = 0
        node_i = initial
        queue = [(0, node_i)]

        while queue:
            weight, node_a = heapq.heappop(queue)

            if node_a not in visited:
                visited.add(node_a)
                minimum_spanning_tree.append((node_a.get_name(), weight))
                total_weight += weight

                if node_a in self.graph_dict:
                    for neighbor, weight_b in self.graph_dict[node_a]:
                        if neighbor not in visited:
                            heapq.heappush(queue, (weight_b, neighbor))
        return minimum_spanning_tree, total_weight

    def kruskal(self):
        edges = []
        for vertex, neighbors in self.graph_dict.items():
            for neighbor, weight in neighbors:
                edges.append((weight, vertex, neighbor))

        edges.sort(key=lambda edge: edge[0])

        parent = {vertex: vertex for vertex in self.graph_dict}

        def find(v):
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]

        def union(v1, v2):
            root1 = find(v1)
            root2 = find(v2)
            parent[root1] = root2

        minimum_spanning_tree = []
        total_weight = 0

        for edge in edges:
            weight, vertex_1, vertex_2 = edge
            if find(vertex_1) != find(vertex_2):
                union(vertex_1, vertex_2)
                minimum_spanning_tree.append((vertex_1.get_name(), vertex_2.get_name(), weight))
                total_weight += weight

        return minimum_spanning_tree, total_weight

    def floyd_warshall(self):
        num_vertices = len(self.graph_dict)
        distance = [[float('inf')] * num_vertices for _ in range(num_vertices)]

        for vertex, neighbors in self.graph_dict.items():
            vertex_index = list(self.graph_dict.keys()).index(vertex)
            distance[vertex_index][vertex_index] = 0
            for neighbor, weight in neighbors:
                neighbor_index = list(self.graph_dict.keys()).index(neighbor)
                distance[vertex_index][neighbor_index] = weight

        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if distance[i][k] + distance[k][j] < distance[i][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]

        print('Distancia más corta entre cada par de nodos:')
        for i in range(num_vertices):
            print("\n-------------------------------------------------------------------")
            for j in range(num_vertices):
                if distance[i][j] == float('inf'):
                    print(f"{list(self.graph_dict.keys())[i]} ---> {list(self.graph_dict.keys())[j]}: INF", end='  | ')
                else:
                    print(f"{list(self.graph_dict.keys())[i]} ---> {list(self.graph_dict.keys())[j]}: {distance[i][j]}", end='  | ')
            print()
        print("\n-------------------------------------------------------------------")
        return distance

    def greedy_coloring(self):
        if self.directed:
            return print("El algoritmo solo se puede ejecutar en grafos no dirigidos")

        color_counter = 0

        for vertex in self.graph_dict:
            neighbor_colors = set()
            for neighbor, _ in self.graph_dict[vertex]:
                neighbor_color = self.colors.get(neighbor)
                if neighbor_color is not None:
                    neighbor_colors.add(neighbor_color)

            vertex_color = None
            for color in range(color_counter):
                if color not in neighbor_colors:
                    vertex_color = color
                    break

            if vertex_color is None:
                vertex_color = color_counter
                color_counter += 1

            self.colors[vertex] = vertex_color

        numero_cromatico = color_counter

        print("Número cromático:", numero_cromatico)
        print("Colores asignados:")
        for vertex, color in self.colors.items():
            print(f"{vertex.get_name()}: {color}")

        return numero_cromatico, self.colors

    def input_vertices(self):
        num_vertices = int(input("Ingrese el número de vértices: "))
        for _ in range(num_vertices):
            vertex_name = input("Ingrese el nombre de un vértice: ")
            vertex = Vertex(vertex_name)
            self.add_vertex(vertex)

    def input_edges(self):
        num_edges = int(input("Ingrese el número de aristas: "))
        for _ in range(num_edges):
            vertex_name_1 = input("Ingrese el nombre del primer vértice de la arista: ")
            vertex_name_2 = input("Ingrese el nombre del segundo vértice de la arista: ")
            weight = float(input("Ingrese el peso de la arista: "))
            vertex_1 = Vertex(vertex_name_1)
            vertex_2 = Vertex(vertex_name_2)
            edge = Edge(vertex_1, vertex_2, weight)
            self.add_edge(edge)

# Programa principal
g = Graph(directed=False)

g.input_vertices()
g.input_edges()

print("\nGrafo:")
print(g)
