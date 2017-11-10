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

    def __init__(self, G):
        super(RandomWalk, self).__init__(G)

    def random_walk(self, start, length, simple=False, probabilities=None, path=None):
        if simple:
            walk = self._simple_random_walk(start, length, probabilities, path)
        else:
            walk = self._random_walk(start, length, probabilities, path)
        return (walk, length)

    def _random_walk(self, start, length, probabilities=None, path=None):

        if probabilities is not None and len(probabilities) != self.N:
            u.printf('probabilities vector does not contain probabilities for all {} nodes.'.format(self.N))
            sys.exit(0)

        if start not in self.states:
            u.printf('{} does not exist in graph.'.format(start))
            sys.exit(0)

        i = self.states.index(start)

        if path is None:
            path = []
        path.append(start)

        row = self.A.getrow(i)
        if row.nnz == 0:
            return path

        outlinks = row.nonzero()[1]
        if probabilities is not None:
            prob = probabilities[outlinks]
            next_id = np.random.choice(outlinks, p=prob / prob.sum())
        else:
            next_id = np.random.choice(outlinks)
        next = self.states[next_id]

        if len(path) < length:
            self._random_walk(next, length, probabilities, path)

        return path

    def _simple_random_walk(self, start, length, probabilities=None, path=None):

        if probabilities is not None and len(probabilities) != self.N:
            u.printf('probabilities vector does not contain probabilities for all {} nodes.'.format(self.N))
            sys.exit(0)

        if start not in self.states:
            u.printf('{} does not exist in graph.'.format(start))
            sys.exit(0)

        i = self.states.index(start)

        if path is None:
            path = []

        if start not in path:
            path.append(start)

        row = self.A.getrow(i)
        if row.nnz == 0:
            return path

        outlinks = row.nonzero()[1]
        if probabilities is not None:
            prob = probabilities[outlinks]
            next_id = np.random.choice(outlinks,p=prob/prob.sum())
        else:
            next_id = np.random.choice(outlinks)
        next = self.states[next_id]

        if len(path) < length:
            if next in path:
                if row.nnz > 1:
                    self._simple_random_walk(start, length, probabilities, path)
                else:
                    return path
            else:
                self._simple_random_walk(next, length, probabilities, path)

        return path

    def random_walks(self, start_points, lengths, simple=False, probabilities=None):
        if len(start_points) != len(lengths):
            u.printf('there should be the same # of elements in start_points and lengths')
            sys.exit(0)
        njobs = -1
        results = Parallel(n_jobs=njobs)(delayed(unwrap_random_walk)((self,start,length,simple,probabilities)) for start,length in zip(start_points, lengths))
        return results

