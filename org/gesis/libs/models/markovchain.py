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
from sklearn.preprocessing import normalize


########################################################################################
# Class
########################################################################################
class MarkovChain(Navigation):

    def __init__(self, M, T):
        super(MarkovChain, self).__init__(M, T)
        self.set_nparams(self.M.shape[0] * (self.M.shape[1] - 2))

    def compute_loglikelihood(self):
        super(MarkovChain, self).compute_loglikelihood()

        ### log-likelihood        
        P = normalize(self.T,'l1',axis=1)
        self.loglikelihood = self.__loglikelihood__(P) 
        