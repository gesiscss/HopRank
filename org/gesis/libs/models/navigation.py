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
from org.gesis.libs.utils import read_csv
from org.gesis.libs.utils import read_graph
from org.gesis.libs.utils import read_series
from org.gesis.libs.utils import read_sparse_matrix
from org.gesis.libs.utils import save_graph
from org.gesis.libs.utils import save_series
from org.gesis.libs.utils import save_sparse_matrix
from org.gesis.libs.utils import to_symmetric

########################################################################################
# System Dependencies
########################################################################################
import os
import gc
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import mmread
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

########################################################################################
# Constants
########################################################################################
ONTO_EXT = '.csv.gz'
COMPRESSION = 'gzip'
ONTO_FN = '<LCC><ONTO>_<YEAR>.<EXT>'
GRAPH_EXT = 'gpickle'
ADJ_EXT = 'mtx'
CSV_EXT = 'csv'
SPECIAL_CONCEPT_NAME = '/ontology/STY/'.lower()
ROOT_CONCEPT_FULL = 'http://www.w3.org/2002/07/owl#Thing'.lower()
ROOT_CONCEPT = 'owl#Thing'.lower()
SPECIAL_CASES = {'crisp': 'csp', 'who-art': 'who', 'chmo': 'obo', 'costart': 'cst', 'loinc': 'lnc', 'tao': 'obo'}


########################################################################################
# Class
########################################################################################
class Navigation(object):

    def __init__(self, M, T):
        '''
        :param M: structure adj. matrix 
        :param T: transitions adj. matrix
        '''
        self.M = M
        self.T = T
        self.params = None
        self.nparams = None
        self.nobservations = None
        self.loglikelihood = None
        self.N = self.M.shape[0]
        self.aic = None
        self.bic = None

    ################################################
    # Getters and Setters
    ################################################
    def set_nparams(self, nparams):
        self.nparams = nparams

    ################################################
    # Evaluation
    ################################################
    def __loglikelihood__(self, P):
        return (self.T.toarray() * np.log(P.toarray())).sum()

    def AIC(self):
        if self.nobservations - self.nparams - 1 == 0:
            return
        if self.nparams == 0 or self.nobservations / self.nparams > 40:
            self.aic = (-2 * self.loglikelihood) + (2 * self.nparams) + ((2 * self.nparams * (self.nparams + 1)) / (self.nobservations - self.nparams - 1))
        else:
            self.aic = (-2 * self.nloglikelihood) + (2 * self.nparams)

    def BIC(self):
        if self.nobservations - self.nparams - 1 == 0:
            return
        self.bic = (-2 * self.loglikelihood) + (self.nparams * np.log(self.nobservations))

    ################################################
    # Abstract methods
    ################################################
    def compute_loglikelihood(self):
        return