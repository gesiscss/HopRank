__author__ = "Lisette Espin-Noboa"
__copyright__ = "Copyright 2018, HopRank"
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
ONTO_FN = '<ONTO>_<YEAR>.<EXT>'
GRAPH_EXT = 'gpickle'
ADJ_EXT = 'mtx'
CSV_EXT = 'csv'
SPECIAL_CONCEPT_NAME = '/ontology/STY/'.lower()
ROOT_CONCEPT_FULL = 'http://www.w3.org/2002/07/owl#Thing'.lower()
ROOT_CONCEPT = 'owl#Thing'.lower()
SPECIAL_CASES = {'crisp':'csp', 'who-art':'who', 'chmo':'obo', 'costart':'cst', 'loinc':'lnc', 'tao':'obo'}

########################################################################################
# Class
########################################################################################
class Ontology(object):
    
    def __init__(self, name, year=None, submission_id=None, root_folder=None):
        self.name = name
        self.year = str(year)
        self.submission_id = submission_id
        self.number_of_nodes = 0
        self.numer_of_edges = 0
        self.G = None
        self.A = None
        self.sorted_nodes = None
        self.root_folder = root_folder
        self.set__path()
    
    ################################################
    # Getters and Setters
    ################################################
    def get_graph(self):
        return self.G
    
    def get_adjacency_matrix(self):
        return self.A
    
    def get_nodes(self):
        return self.sorted_nodes
    
    def get_onto_filename(self, path, ext):
        return ONTO_FN.replace('<ONTO>',self.name).replace('<YEAR>',self.year).replace('<EXT>',ext)
        
    def set_root_folder(self, path):
        self.root_folder = path
        self.set__path()

    def set__path(self):
        if self.submission_id is None:
            raise ValueError("submission_id has not been loaded!")
        if self.root_folder is None:
            self._path = None
        else:
            self._path = os.path.join(self.root_folder,self.name,str(self.submission_id))
        
    ################################################
    # Methods
    ################################################
    
        
    def load_ontology(self):
        fn = [fn for fn in os.listdir(self._path) if fn.startswith(self.name) and fn.endswith(ONTO_EXT)]
        if len(fn) == 0:
            raise ValueError("Ontology file not found in {}".format(self._path))
        try:
            fn = os.path.join(self._path,fn[0])
            df = read_csv(fn,index_col=False,compression=COMPRESSION)
            printf('{} loaded!'.format(fn))            
        except Exception as ex:
            printf(ex)
            printf('ERROR: {}-{} NOT loaded!'.format(self.name,self.year))
            return
        self._convert_DataFrame_to_DiGraph(df)  
        self._sort_nodes()
        
    def create_adjacency_matrix(self):
        if self.G is None:
            raise ValueError("Ontology graph has not been loaded!")
        self.A = nx.to_scipy_sparse_matrix(self.G, nodelist=self.sorted_nodes)
        
    def _convert_DataFrame_to_DiGraph(self, df):
        columns = ['Class ID', 'Parents']
        try:
            tmp = df[columns]
            edges = [(get_short_concept_name(self.name,parent),get_short_concept_name(self.name,row['Class ID'])) for index,row in tmp.iterrows() for parent in str(row['Parents']).split('|') ]
            self.G = nx.DiGraph()
            self.G.graph['name'] = self.name
            self.G.graph['year'] = self.year
            self.G.graph['submission_id'] = self.submission_id
            self.G.add_edges_from(edges)
            
            try:
                self.G.remove_nodes_from(['nan'])
            except:
                pass
            
            printf('Convertion {}-{} DataFrame to DiGraph done!'.format(self.name,self.year))
        except Exception as ex:
            printf(ex)
            printf('ERROR: {}-{} NOT converted from DataFrame to DiGraph!'.format(self.name,self.year))
        
    def _sort_nodes(self):
        try:
            self.sorted_nodes = sorted(list(self.G.nodes()))
        except Exception as ex:
            printf(ex)
            printf('ERROR: {}-{} NOT sorted nodes!'.format(self.name,self.year))

            
    def create_hops_matrices(self, path, maxk=5):
        
        if self.A is None:
            printf('{}-{}-{}: Adjacency matrix is not loaded.'.format(self.name, self.year, self.submission_id))
            return
        
        fn = self.get_khop_matrix_fn()
        kdone = None
        hop = None
        shape = None
        for k in range(1,maxk+1,1):

            if os.path.exists(os.path.join(path,fn.replace('<k>',str(k)))):
                printf('=== {}-{}: {}HOP already exists (pass)'.format(self.name, self.year, k))
                kdone = k
                continue

            printf('=== {}-{}: {}HOP ==='.format(self.name,self.year,k))
            if k == 1:
                hop = self.A.copy()
                shape = hop.shape
            else:

                if hop is None:
                    hop = self.get_khop_matrix(path, kdone)
                    shape = hop.shape

                if hop.sum() == 0:
                    printf('the matrix has already reached zero (break). Up to {}HOP'.format(k - 1))
                    break

                hop = csr_matrix(hop.dot(self.A))
                # printf('1. dot product')

                hop = sparse.find(hop)
                # printf('2. >0')

                hop = csr_matrix((np.ones(hop[2].size).astype(np.int8), (hop[0], hop[1])), shape, dtype=np.int8)
                # printf('3. >0 --> 1')

                hop = hop.tolil()
                # printf('4. to lil')

                hop.setdiag(0)
                # printf('5. diagonal zero')

                hop = hop.tocsr()
                # printf('6. csr')

                hop.eliminate_zeros()
                # printf('6. eliminate zeros')

                if hop.sum() > 0:

                    # removing previous HOPS
                    for previous_k in range(k - 1, 0, -1):

                        previous_hop = self.get_khop_matrix(path, previous_k)                         
                        # printf('9. loaded previous k done: {}'.format(previous_k))

                        hop = hop - previous_hop
                        # printf('10. minus')

                        hop = (hop > 0).astype(np.int8)
                        # printf('11. >0')

                        hop = hop.tolil()
                        # printf('12. to lil')

                        hop.setdiag(0)
                        # printf('13. diagonal to 0')

                        hop = hop.tocsr()
                        # printf('14. to csr')

                        hop.eliminate_zeros()
                        # printf('15. eliminate zeros')

                        if hop.sum() == 0:
                            printf('the matrix has already reached zero (break). Up to {}HOP'.format(previous_k))
                            break

                else:
                    printf('the matrix has already reached zero (break). Up to {}HOP'.format(k-1))
                    break

            printf('saving {}-{} {}hop...'.format(self.name,self.year,k))
            comment = 'k-hop:{}\nOntology: {}\nYear: {}\nSubmissionID: {}'.format(k,self.name, self.year, self.submission_id)
            field = 'integer'
            save_sparse_matrix(hop, path, fn.replace('<k>',str(k)), comment, field)
            printf('{}-{} {}hop saved! --> {} shape, {} sum'.format(self.name, self.year, k, hop.shape, hop.sum()))
            kdone = k

        gc.collect()
        printf('=== {}-{}: done for {} HOPs! ==='.format(self.name, self.year, kdone))        
        return 0 if kdone == 1 and hop.sum() == 0 else kdone
        
    ################################################
    # I/O
    ################################################
    def load_graph(self, path):
        self.G = read_graph(path, self.get_onto_filename(path,GRAPH_EXT))
        
    def save_graph(self, path):
        if self.G is None:
            raise ValueError("Ontology graph has not been loaded!")
        save_graph(self.G, path, self.get_onto_filename(path,GRAPH_EXT))

    def get_khop_matrix_fn(self):
        return '{}_{}_<k>HOP.mtx'.format(self.name, self.year)
    
    def get_khop_matrix(self, path, k):
        fn = self.get_khop_matrix_fn()
        return read_sparse_matrix(path, fn.replace('<k>', str(k)))
        
        
    def load_adjacency(self, path):
        comment = 'Ontology: {}\nYear: {}\nSubmissionID: {}'.format(self.name, self.year, self.submission_id)
        field = 'integer'
        self.A = read_sparse_matrix(path, self.get_onto_filename(path,ADJ_EXT))
        
    def save_adjacency(self, path):
        if self.A is None:
            raise ValueError("Ontology adj. matrix has not been loaded!")
        comment = 'Ontology: {}\nYear: {}\nSubmissionID: {}'.format(self.name, self.year, self.submission_id)
        field = 'integer'
        save_sparse_matrix(self.A, path, self.get_onto_filename(path,ADJ_EXT), comment=comment, field=field)
    
    def load_nodes(self, path):        
        self.sorted_nodes = read_series(path, self.get_onto_filename(path,CSV_EXT))
        
    def save_nodes(self, path):
        if self.sorted_nodes is None:
            raise ValueError("Sorted nodes has not been loaded!")
        save_series(pd.Series(self.sorted_nodes), path, self.get_onto_filename(path,CSV_EXT))
        
        
########################################################################################
# Functions
########################################################################################     
def get_short_concept_name(ontology_name,concept_name):
    # http://identifiers.org/mamo/MAMO_0000037
    # http://purl.bioontology.org/ontology/csp/0027-1585
    
    concept_name = concept_name.lower()
    ontology_name = ontology_name.lower()
    
    if ROOT_CONCEPT == concept_name:
        scn = concept_name
    
    elif ROOT_CONCEPT_FULL == concept_name:
        scn = ROOT_CONCEPT
    
    elif SPECIAL_CONCEPT_NAME in concept_name:
        scn = '{}/{}'.format(SPECIAL_CONCEPT_NAME.split('/')[2], concept_name.split(SPECIAL_CONCEPT_NAME)[-1])

    else:
        if ontology_name in SPECIAL_CASES:
                scn = concept_name.split("/{}/".format(SPECIAL_CASES[ontology_name]))[-1]
        else:
            scn = concept_name.split("/{}/".format(ontology_name))[-1]

    return scn.lower()

def main():
    printf('class ontology')
          
if __name__ == '__main__':
    main()