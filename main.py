print ("Esto funciona main.py")

class Vertex:
    def __init__(self):
        self.visitado = False
        # self.padre = None # para el DFS
        # self.nivel = -1# para el BFS
        self.vecinos = []#relacion aristas adyacentes
        
    def Agrega_vecinos(self, v):
        if not v in self.vecinos:
            self.vecinos.append(v)

# class Edge:
#     #Constructor
#     def __init__(self, vertex_1, vertex_2):
#         self.vertex_1 = vertex_1
#         self.vertex_2 = vertex_2

#     # Obtener datos de la clase
#     def __str__(self):
#         return self.vertex_1.get_name() + " ---> " + self.vertex_2.get_name()
    
    

#     #Gets and sets
#     def get_vertex_1(self):
#         return self.vertex_1
#     def get_vertex_2(self):
#         return self.vertex_2

class Graph:
    
    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex):
        if not vertex in self.graph:
            self.graph[vertex] = Vertex(vertex)

    def Agrega_arista(self,a,b):#a y b son verticex s0
        if a in self.graph and b in self.graph:#si estan en la lista agrega la arista
            self.graph[a].Agrega_vecinos(b)
            self.graph[b].Agrega_vecinos(a)

    def bfs(self, start):
        visited = []
        queue = [] 
        visited.append(start)
        queue.append(start)

        while queue:
            s = queue.pop(0) #desencolado
            print (s, end = " ") 

            for neighbour in self.graph[s]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)

# Programa principal


g = Graph()
v = [1,2,3,4,5,6,7,8]
for i in v:
    g.add_vertex(i)

g.add_vertex(1)
g.add_vertex(1)


