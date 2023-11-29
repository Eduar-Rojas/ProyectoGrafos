print ("Hola mundo")

import queue 
import networkx as nx
import matplotlib.pyplot as plt
import heapq

class Grafo:
    def __init__(self):
        self.vertices = []
        self.aristas = []
    def agrega(self,nodos,aristas):
        self.vertices = nodos
        self.aristas = aristas
        self.grafo = {nodo: [] for nodo in nodos}
        for arista in aristas:
            nodo_origen, nodo_destino = arista
            self.grafo[nodo_origen].append(nodo_destino)
        print("grafo: ",self.grafo)
        return

    def dfs(self,nodoI):
        visitados = []
        def dfs_r(nodo):
            print(nodo,end=" ")
            visitados.append(nodo)
            for vecino in self.grafo[nodo]:
                if vecino not in visitados:
                    dfs_r(vecino)
        dfs_r(nodoI)
        print(visitados,"vis")
        return visitados
    def bfs(self,nodoI):
        cola = queue.Queue()
        visitados = []
        
        cola.put(nodoI)
        print("encoladno1:")
        visitados.append(nodoI)
        while not cola.empty():#mientra cola no este vacia
            nodocurrent = cola.get()#desencolo
            print("desencolando", nodocurrent)
            
            for vecino in self.grafo[nodocurrent]:#veo los nodos adyacentes al actual nodo desencolado
                if vecino not in visitados:#si encuentra un nodo que no este marcado como visitado
                    print("encolando",vecino,"\n")
                    cola.put(vecino)#lo agrega a la cola

                    visitados.append(vecino) #y marca como visitado

        print(visitados,"vis bfs")
        return visitados


nodos = [1,2,3,4,5,6]
aristas = [(1,4),(1,3),(2,4),(2,6),(3,2),(3,5),(4,2),(4,6),(5,2),(6,2)]

gr = Grafo()


gr.agrega(nodos,aristas)





def visualizarGrafo(nodos,aristas):
    vG = nx.DiGraph()
    vG.add_nodes_from(nodos)
    vG.add_edges_from(aristas)
    pos =  nx.spring_layout(vG)
    nx.draw(vG, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='lightblue', arrows= True)   
    plt.show()


