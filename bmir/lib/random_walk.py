import networkx as nx
import sys
import bmir.utils as u
import numpy as np
from joblib import Parallel, delayed
from itertools import combinations, permutations
from bmir.lib.graph_matrix import GraphMatrix

def unwrap_random_walk(arg):
    np.random.seed(None)
    return RandomWalk.random_walk(arg[0],arg[1],arg[2],arg[3],arg[4],None)

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
