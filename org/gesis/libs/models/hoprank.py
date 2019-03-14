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
class HopRank(Navigation):

    def __init__(self, M, T, khops_fnc=None, betas=None, khops=None):
        super(HopRank, self).__init__(M, T)
        self.__validate__(khops_fnc, betas, khops)
        self.set_nparams(len(betas))
        self.betas = betas
        self.khops_fnc = khops_fnc
        self.khops = khops

    def __validate__(self, khops_fnc, betas, khops):
        if khops_fnc is None and khops is None:
            raise ValueError("either khops_fnc callback or khops matrix must exist.") # check what kop do i need        
        if betas is None:
            raise ValueError("betas vector must exist.")

    def compute_loglikelihood(self):
        super(HopRank, self).compute_loglikelihood()

        P = None
        for hop,row in self.betas.iterrows():
            beta = row.beta
            if hop == 0:
                P = csr_matrix(np.ones((self.N, self.N)) * (beta / self.N))
            else:
                # DO NOT DO TOARRAY
                if self.khops_fnc is not None:
                    m = self.khops_fnc(hop)
                else:
                    m = csr_matrix(np.isin(self.khops.toarray(), [hop], assume_unique=False).astype(np.int8))                
                P += csr_matrix(beta * normalize(m, norm='l1', axis=1))

        P = normalize(P, norm='l1', axis=1)
        self.loglikelihood = self.__loglikelihood__(P)
        return P
