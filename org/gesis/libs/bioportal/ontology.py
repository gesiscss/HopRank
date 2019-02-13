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
from org.gesis.libs.utils import to_symmetric
from org.gesis.libs.utils import get_khop_with_partial_results

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
ADJ_EXT = 'npz'
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
        self.uA = None
        self.lcc_A = None
        self.lcc_uA = None
        self.sorted_nodes = None
        self.lcc_sorted_nodes = None
        self.root_folder = root_folder
        self.lcc = None
        self.path_khop = None
        self.set__path()
    
    ################################################
    # Getters and Setters
    ################################################
    def get_graph(self):
        return self.G
    
    def get_adjacency_matrix(self, lcc=False):
        if lcc:
            return self.lcc_A
        return self.A
    
    def get_undirected_adjacency(self, lcc=False):
        if lcc:
            if self.lcc_uA is None:
                self.lcc_uA = to_symmetric(self.lcc_A)
            return self.lcc_uA

        if self.uA is None:
            self.uA = to_symmetric(self.A)
        return self.uA
    
    def get_nodes(self, lcc=False):
        if lcc:
            return self.lcc_sorted_nodes
        return self.sorted_nodes
    
    def get_onto_filename(self, path, ext, lcc=False):
        return ONTO_FN.replace('<LCC>','LCC_' if lcc else '').replace('<ONTO>',self.name).replace('<YEAR>',self.year).replace('<EXT>',ext)
    
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
    
    def set_lcc(self, lcc=False):
        self.lcc = lcc
        
    def set_path_khop(self, path):
        self.path_khop = path
        
    def get_khop(self, k):
        fn = self.get_khop_matrix_fn(k, lcc=self.lcc)
        return read_sparse_matrix(self.path_khop, fn)
    
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
        self.sorted_nodes = sorted(list(self.G.nodes()))
        self.lcc_sorted_nodes = sorted(list(max(nx.connected_component_subgraphs(self.G.to_undirected()), key=len).nodes()))
    
    
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
    
    def create_adjacency_matrix(self, lcc=False):
        if self.G is None:
            raise ValueError("Ontology graph has not been loaded!")
            
        if lcc:
            self.lcc_A = nx.to_scipy_sparse_matrix(self.G, nodelist=self.lcc_sorted_nodes, format='csr')
        else:
            self.A = nx.to_scipy_sparse_matrix(self.G, nodelist=self.sorted_nodes, format='csr')
                   
    
    def create_hops_matrices(self, path, maxk=5, lcc=False):
        
        reached_zero = False
        
        if lcc:
            if self.lcc_A is None:
                printf('{}-{}-{}: Adjacency matrix is not loaded.'.format(self.name, self.year, self.submission_id))
                return
            A = self.lcc_A
        else:
            if self.A is None:
                printf('{}-{}-{}: Adjacency matrix is not loaded.'.format(self.name, self.year, self.submission_id))
                return
            A = self.A
        
        uA = self.get_undirected_adjacency(lcc).tocsr() # undirected      
        kdone = 1
                
        khops = get_khop_with_partial_results(uA,maxk)
        for k,hop in khops:    
            
            if hop.sum() == 0:
                printf('{}-{}-{}: {}-hop has reached zero!'.format(self.name, self.year, self.submission_id, k))
                break

            kdone = k
            
            # save            
            printf('{}-{}-{}: {}-hop --> shape:{}, sum:{}!'.format(self.name, self.year, self.submission_id, k, hop.shape, hop.sum())) 
            printf('{}-{}-{}: {}-hop saving...'.format(self.name, self.year, self.submission_id, k))
            
            fn = self.get_khop_matrix_fn(k, lcc=lcc)
            save_sparse_matrix(hop, path, fn)
            printf('{}-{}-{}: {}-hop done!'.format(self.name, self.year, self.submission_id, k))
            printf('')            
            
        return kdone

    
    def create_distance_matrix(self, path, hopspath, lcc=False):    
                
        fn_final = self.get_distance_matrix_fn(lcc)
        if os.path.exists(os.path.join(path, fn_final)):
            return read_sparse_matrix(path, fn_final)
                                      
        fname = self.get_khop_matrix_fn(k=None,lcc=lcc)
        files = [fn for fn in os.listdir(hopspath) if fn.startswith(fname.replace('<k>HOP.{}'.format(ADJ_EXT),'')) and fn.endswith('HOP.{}'.format(ADJ_EXT))]
        m = None
        
        print('\n'.join(files))
        maximun = len(files)
        
        for fn in files:    
            khop = int(fn.split('_')[-1].split('HOP')[0])            
            
            print('file:{}, k:{}'.format(fn,khop))
            
            if m is None:
                m = read_sparse_matrix(hopspath, fn) * khop
            else:
                m += read_sparse_matrix(hopspath, fn) * khop
            m.eliminate_zeros()
            
            if m.max() > maximun:
                print('>>> weird: {} in k:{} in {}'.format(m.max(),khop,self.name))
                return None
        
        #m = m.tolil()
        #m.setdiag(0)
        #m = m.tocsr()
        m.eliminate_zeros()
        
        comment = 'LCC: {}\nOntology: {}\nYear: {}\nSubmissionID: {}'.format(lcc,self.name, self.year, self.submission_id)
        field = 'integer'
        save_sparse_matrix(m, path, fn_final, comment=comment, field=field)        
        return m
    
    ################################################
    # I/O
    ################################################
    def load_graph(self, path):
        self.G = read_graph(path, self.get_onto_filename(path,GRAPH_EXT))
        
    def save_graph(self, path):
        if self.G is None:
            raise ValueError("Ontology graph has not been loaded!")
        save_graph(self.G, path, self.get_onto_filename(path,GRAPH_EXT))

    def get_khop_matrix_fn(self, k=1, lcc=False):    
        if k is not None:
            return '{}{}_{}_{}HOP.{}'.format('LCC_' if lcc else '', self.name, self.year, k, ADJ_EXT)  
        return '{}{}_{}_<k>HOP.{}'.format('LCC_' if lcc else '', self.name, self.year, ADJ_EXT)  
    
    def get_distance_matrix_fn(self, lcc=False):
        return '{}{}_{}_HOPs.{}'.format('LCC_' if lcc else '', self.name, self.year, ADJ_EXT)
    
    def get_khop_matrix(self, path, k, lcc=False):
        fn = self.get_khop_matrix_fn(k, lcc)
        return read_sparse_matrix(path, fn).tocsr()
        
        
    def load_adjacency(self, path, lcc=False):
        comment = 'Ontology: {}\nYear: {}\nSubmissionID: {}'.format(self.name, self.year, self.submission_id)
        field = 'integer'
        if lcc:
            self.lcc_A = read_sparse_matrix(path, self.get_onto_filename(path,ADJ_EXT,lcc))
        else:
            self.A = read_sparse_matrix(path, self.get_onto_filename(path,ADJ_EXT))
        
    def save_adjacency(self, path, lcc=False):
        if self.A is None:
            raise ValueError("Ontology adj. matrix has not been loaded!")
        comment = 'LCC: {}\nOntology: {}\nYear: {}\nSubmissionID: {}'.format(lcc,self.name, self.year, self.submission_id)
        field = 'integer'
        if lcc:
            save_sparse_matrix(self.lcc_A, path, self.get_onto_filename(path,ADJ_EXT,lcc), comment=comment, field=field)
        else:
            save_sparse_matrix(self.A, path, self.get_onto_filename(path,ADJ_EXT), comment=comment, field=field)
    
    def load_nodes(self, path, lcc=False):        
        if lcc:
            self.lcc_sorted_nodes = read_series(path, self.get_onto_filename(path,CSV_EXT,lcc))
        else:
            self.sorted_nodes = read_series(path, self.get_onto_filename(path,CSV_EXT))
        
    def save_nodes(self, path, lcc=False):
        
        if lcc:
            if self.lcc_sorted_nodes is None:
                raise ValueError("LCC sorted nodes has not been loaded!")
            save_series(pd.Series(self.lcc_sorted_nodes), path, self.get_onto_filename(path,CSV_EXT,lcc))
            
        else:
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