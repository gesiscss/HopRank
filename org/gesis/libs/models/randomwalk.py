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
from org.gesis.libs.utils import printf
from org.gesis.libs.models.navigation import Navigation

########################################################################################
# System Dependencies
########################################################################################
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize


########################################################################################
# Class
########################################################################################
class RandomWalk(Navigation):

    NPARAMS = 0

    def __init__(self, M, T, alpha=None):
        super(RandomWalk, self).__init__(M, T)
        self.alpha = self.__get_damping_factor__(alpha)
        self.__validate__()
        self.set_nparams(self.NPARAMS)

    def __validate__(self):
        if self.alpha is not None and (self.alpha < 0 or self.alpha > 1):
            raise ValueError("damping factor must be between 0 and 1 (inclusive).")
            return

    def __get_damping_factor__(self, alpha):
        if alpha is None:
            alpha = round(self.M.multiply(self.T).sum() / self.T.sum(), 2)
            printf('Empirical alpha (damping factor): {}'.format(alpha))
            return alpha
        return alpha

    def compute_loglikelihood(self):
        super(RandomWalk, self).compute_loglikelihood()

        if self.alpha > 0:
            if self.alpha == 1.0:
                alpha = 0.99999

            ### random walk
            P = lil_matrix(self.alpha * normalize(self.M, norm='l1', axis=1))
            T = lil_matrix((np.ones((self.N, self.N)) * (1 - self.alpha)) / self.N)
            P = lil_matrix(P + T)
        else:
            ### always random jump (teleportation)
            P = lil_matrix(np.ones((self.N, self.N)) / self.N)

        ### log-likelihood
        self.loglikelihood = self.__loglikelihood__(P)

