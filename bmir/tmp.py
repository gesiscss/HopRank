from bmir.lib.random_walk import RandomWalk
from bmir.lib.page_rank import PageRank
import networkx as nx
from joblib import Parallel, delayed
import numpy as np

G = nx.DiGraph()
G.add_edge('A','B')
G.add_edge('A','C')
G.add_edge('A','D')
G.add_edge('B','E')
G.add_edge('B','F')
G.add_edge('C','G')
G.add_edge('C','H')
G.add_edge('D','I')
G.add_edge('D','J')
G.add_edge('E','K')

G = G.to_undirected()
RW = RandomWalk(G)
start_points = ['A','A','A','A','B','B','B','B','H','H']
lengths = [2,3,4,5,2,3,4,5,5,4]

### UNIFORM RANDOM WALK
print('===== UNIFORM RANDOM WALK =====')
for simple in [True,False]:
    print('\n- Simple:{}'.format(simple))
    results = RW.random_walks(start_points,lengths,simple,None)
    walks = filter(lambda a: len(a[0]) == a[1], results)
    residuals = filter(lambda a: len(a[0]) != a[1], results)
    paths = [w[0] for w in walks]
    print walks

### PAGERANK AND RANDOM WALK
print('===== PAGERANK RANDOM WALK =====')
PR = PageRank(G)
PR.run()
probabilities = PR.probabilities()
print(probabilities)
for simple in [True,False]:
    print('\n- Simple:{}'.format(simple))
    results = RW.random_walks(start_points,lengths,simple,probabilities)
    walks = filter(lambda a: len(a[0]) == a[1], results)
    residuals = filter(lambda a: len(a[0]) != a[1], results)
    paths = [w[0] for w in walks]
    print walks