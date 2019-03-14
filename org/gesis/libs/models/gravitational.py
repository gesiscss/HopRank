__author__ = "Lisette Espin-Noboa"
__copyright__ = "Copyright 2019, HopRank"
__credits__ = ["Florian Lemmerich", "Markus Strohmaier", "Simon Walk", "Mark Musen"]
__license__ = "GPL"
__version__ = "1.0.3"
__maintainer__ = "Lisette Espin-Noboa"
__email__ = "Lisette.Espin@gesis.org"
__status__ = "Developing"

########################################################################################
# Local Dependencies
########################################################################################
from org.gesis.libs.models.navigation import Navigation

########################################################################################
# System Dependencies
########################################################################################
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


########################################################################################
# Class
########################################################################################
class Gravitational(Navigation):

    NPARAMS = 0

    def __init__(self, M, T, distances):
        super(Gravitational, self).__init__(M, T)
        self.set_nparams(self.NPARAMS)
        self.distances = distances
        self.__validate__()

    def __validate__(self):
        if self.distances is None:
            raise ValueError("distances matrix must exist.") # check what kop do i need

    def compute_loglikelihood(self):
        super(Gravitational, self).compute_loglikelihood()

        distance = self.distances.toarray()
        distance[distance == 0] = distance.max() + 1
        np.fill_diagonal(distance, distance.max() + 1)
        distance = 1 / (distance.astype(np.float64)) ** 2

        # gravitational
        P = csr_matrix(np.multiply(np.repeat(self.M.sum(axis=0), self.N, axis=0), distance))        
        P += csr_matrix(np.ones((self.N, self.N)))

        # norm
        P = normalize(P, norm='l1', axis=1)
        self.loglikelihood = self.__loglikelihood__(P)
