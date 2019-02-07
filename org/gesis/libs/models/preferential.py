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
class PreferentialAttachment(Navigation):

    NPARAMS = 0

    def __init__(self, M, T):
        super(PreferentialAttachment, self).__init__(M, T)
        self.set_nparams(self.NPARAMS)

    def compute_loglikelihood(self):
        super(PreferentialAttachment, self).compute_loglikelihood()

        # pref. attachment)
        P = lil_matrix(np.repeat(self.M.sum(axis=0), self.N, axis=0))
        P += lil_matrix(np.ones((self.N, self.N)) * (1 / self.N))  # maybe not

        # norm
        P = normalize(P, norm='l1', axis=1)
        self.loglikelihood = self.__loglikelihood__(P)
