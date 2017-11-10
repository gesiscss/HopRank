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
length = 1000
teleportation = 0.15
np.set_printoptions(precision=2)

#############################################################################
# UNIFORM
#############################################################################

### MODEL 1
print('\n===== M1: UNIFORM - RANDOM WALK WITH TELEPORTATION {} ====='.format(teleportation))
path = RW.random_walk(length,teleportation=teleportation)
print(RW.likelihood(path).toarray())

### MODEL 2
print('\n===== M2: UNIFORM - RANDOM WALK WITHOUT TELEPORTATION =====')
path = RW.random_walk(length)
print(RW.likelihood(path).toarray())

#############################################################################
# PAGERANK
#############################################################################

PR = PageRank(G)
PR.run()
nodebias = PR.probabilities()
print('\nPageRanks: \n{}'.format(nodebias))

### MODEL 3
print('\n===== M3: PAGERANK - RANDOM WALK WITH TELEPORTATION {} ====='.format(teleportation))
path = RW.random_walk(length,teleportation=teleportation,nodebias=nodebias)
print(RW.likelihood(path).toarray())

### MODEL 4
print('\n===== M4: PAGERANK - RANDOM WALK WITHOUT TELEPORTATION =====')
path = RW.random_walk(length,nodebias=nodebias)
print(RW.likelihood(path).toarray())