from bmir.lib.graph_matrix import GraphMatrix
import numpy as np

class PageRank(GraphMatrix):

    def __init__(self, G):
        super(PageRank, self).__init__(G)
        self.rank = None

    def run(self, alpha=0.85, maxerr=0.00001):
        r_previous = np.zeros(self.N)
        self.rank = np.ones(self.N) / self.N
        iteration = 1
        converged = 0

        error = np.sum(np.abs(self.rank - r_previous))
        while (error > maxerr):
            r_previous = self.rank.copy()

            for i, node in enumerate(self.states):
                # inlinks
                Ai = np.array(self.A[:, i].todense())[:, 0]

                # outlinks of each i inlink
                nj = np.array([float(self.A[[j], :].nnz) for j in Ai.nonzero()[0]])

                # pr of i inlinks
                PRj = r_previous * Ai
                PRj = PRj[PRj > 0]

                # right side of eq.
                PRj = (PRj / nj).sum()

                # teleporting
                t = (1 - alpha) / self.N

                # next
                self.rank[i] = t + PRj

            iteration += 1

            tmp = np.sum(np.abs(self.rank - r_previous))
            converged = (converged + 1) if abs(error - tmp) < maxerr else 0
            error = tmp
            if converged == 10:
                break

        return self.rank

    def likelihood(self):
        return self.rank / self.rank.sum()