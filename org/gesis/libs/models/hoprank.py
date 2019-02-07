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
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize


########################################################################################
# Class
########################################################################################
class HopRank(Navigation):

    def __init__(self, M, T, khop, betas):
        super(HopRank, self).__init__(M, T)
        self.__validate__(khop, betas)
        self.set_nparams(len(betas))
        self.betas = betas
        self.khop = khop

    def __validate__(self, khop, betas):
        if khop is None:
            raise ValueError("khop matrix must exist.") # check what kop do i need
        if betas is None:
            raise ValueError("betas vector must exist.")

    def compute_loglikelihood(self):
        super(HopRank, self).compute_loglikelihood()

        P = None
        for hop, beta in enumerate(self.betas):
            if hop == 0:
                P = lil_matrix(np.ones((self.N, self.N)) * (beta / self.N))
            else:
                m = lil_matrix(np.isin(self.khop.toarray(), [hop], assume_unique=False).astype(np.int8))
                P += lil_matrix(beta * normalize(m, norm='l1', axis=1))

        P = normalize(P, norm='l1', axis=1)
        self.loglikelihood = self.__loglikelihood__(P)
