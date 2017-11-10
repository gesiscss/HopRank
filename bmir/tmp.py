from bmir.lib.random_walk import RandomWalk
from bmir.lib.page_rank import PageRank
import networkx as nx
from joblib import Parallel, delayed
import numpy as np
from collections import Counter

#############################################################################
# GRAPH
#############################################################################

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
print(nx.info(G))

#############################################################################
# PARAMS
#############################################################################

RW = RandomWalk(G,True)
length = 110
teleportation = 0.15

#############################################################################
# MODELS - BIASED RANDOM WALKERS
#############################################################################

### MODEL 1
print('===== M1: UNIFORM - RANDOM WALK WITH TELEPORTATION =====')
transitions = RW.random_walk(length,teleportation=teleportation)
print Counter(transitions)

### MODEL 2
print('===== M2: UNIFORM - RANDOM WALK WITHOUT TELEPORTATION =====')
transitions = RW.random_walk(length)
print Counter(transitions)

### MODEL 3
print('===== M3: PAGERANK - RANDOM WALK WITH TELEPORTATION =====')
PR = PageRank(G)
PR.run()
nodebias = PR.probabilities()
print(nodebias.tolist())
transitions = RW.random_walk(length,teleportation=teleportation,nodebias=nodebias)
print Counter(transitions)

### MODEL 4
print('===== M4: PAGERANK - RANDOM WALK WITHOUT TELEPORTATION =====')
PR = PageRank(G)
PR.run()
nodebias = PR.probabilities()
print(nodebias.tolist())
transitions = RW.random_walk(length,nodebias=nodebias)
print Counter(transitions)