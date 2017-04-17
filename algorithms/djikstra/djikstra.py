# Mohammad Saad
# 4/16/2017
# djikstra.py
# Quick implementation of Djikstra's algorithm.


import networkx as nx

G = nx.Graph()

for i in range(0, 9):
    G.add_node(i)

G.add_edge(0, 1, weight=4)
G.add_edge(0, 7, weight=8)
G.add_edge(1,7, weight=11)
G.add_edge(1, 2,weight=8)
G.add_edge(2, 3,weight=7)
G.add_edge(3, 4,weight=9)
G.add_edge(4, 5,weight=10)
G.add_edge(3, 5,weight=14)
G.add_edge(2, 5,weight=4)
G.add_edge(5, 6,weight=2)
G.add_edge(6, 8,weight=6)
G.add_edge(8, 2,weight=2)
G.add_edge(6, 7,weight=1)
G.add_edge(8, 7,weight=7)

print G.edges()

def djikstra(graph):
    pass
