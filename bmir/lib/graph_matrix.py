import networkx as nx
import bmir.utils as u
import sys

class GraphMatrix(object):

    def __init__(self, G):
        self.G = G
        self.A = None
        self.N = self.G.number_of_nodes()
        self.states = None
        self.validate()

    def validate(self):
        if self.G is not None and isinstance(self.G,nx.Graph):
            self.states = self.G.nodes()
            if self.A is None:
                self.A = u.get_adjacency_matrix(self.G,self.states)
                self.G = None
            return True
        u.printf('G must exist and be an instance of networkx.Graph')
        sys.exit(0)

    def likelihood(self):
        return
