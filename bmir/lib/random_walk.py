import networkx as nx
import bmir.utils as u
import numpy as np
from joblib import Parallel, delayed
from itertools import combinations, permutations
from bmir.lib.graph_matrix import GraphMatrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
import sys
sys.setrecursionlimit(100000)

class RandomWalk(GraphMatrix):

    def __init__(self, G, undirected=True):
        super(RandomWalk, self).__init__(G, undirected)

    def random_walk(self, length, teleportation=None, nodebias=None, start=None, path=None):

        if nodebias is not None and len(nodebias) != self.N:
            u.printf('probabilities vector does not contain probabilities for all {} nodes.'.format(self.N))
            sys.exit(0)

        # start node
        if start is None:
            start = np.random.choice(self.states)
        start_id = self.states.index(start)
        next_id = start_id

        if path is None:
            path = []
        path.append(start)

        # teleport (0) or neighbor (1)?
        if teleportation is not None:
            coin = np.random.choice(2,p=[teleportation,1.-teleportation])
        else:
            coin = 1

        if coin == 0:
            # teleportation
            while start_id == next_id:
                next = np.random.choice(self.states)
                next_id = self.states.index(next)
        else:
            # random/biased neighbor
            row = self.A.getrow(start_id)
            if row.nnz == 0:
                return path

            nieghbors = row.nonzero()[1]
            if nodebias is not None:
                # bias
                prob = nodebias[nieghbors]
                next_id = np.random.choice(nieghbors, p=prob / prob.sum())
            else:
                # random
                next_id = np.random.choice(nieghbors)

            next = self.states[next_id]

        # need more walks?
        if len(path) < length:
            self.random_walk(length, teleportation, nodebias, next, path)

        return path

    def likelihood(self, path):
        P = lil_matrix((self.N,self.N))
        source = None
        target = None

        for node in path:
            node_id = self.states.index(node)

            if source is None and target is None:
                source = node_id

            if source is not None and target is None:
                target = node_id

            if source is not None and target is not None:
                P[source,target] += 1
                source = target
                target = None

        return normalize(P.tocsr(), norm='l1', axis=1, copy=False, return_norm=False)