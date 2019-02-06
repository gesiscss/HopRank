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
from org.gesis.libs.utils import read_sparse_matrix
from org.gesis.libs.utils import save_graph
from org.gesis.libs.utils import save_sparse_matrix
from org.gesis.libs.utils import save_series
from org.gesis.libs.bioportal.ontology import get_short_concept_name
from org.gesis.libs.utils import to_symmetric

########################################################################################
# System Dependencies
########################################################################################
import os
import time
import urllib
import datetime
import tldextract
import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import mmread
import scipy.sparse as sparse
from collections import defaultdict
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
CS_FN = '<ONTO>_<YEAR>_<NAVITYPE>.<EXT>'
GRAPH_EXT = 'gpickle'
ADJ_EXT = 'mtx'
CSV_EXT = 'csv'
TMPFOLDER = '/bigdata/lespin/tmp/'
ALLNAVITYPE = 'ALL'
MIN_SESSION_LENGTH = 2

########################################################################################
# Class
########################################################################################
class Transition(object):
    
    def __init__(self, name, navitype, year=None):
        self.name = name
        self.navitype = navitype
        self.year = str(year)
        self.number_of_nodes = 0
        self.numer_of_edges = 0
        self.H = None
        self.T = None
        self.uT = None # undirected T
        self.sorted_nodes = None
        
    ################################################
    # Getters and Setters
    ################################################
    def get_graph(self):
        return self.H
    
    def get_adjacency_matrix(self):
        return self.T
    
    def get_undirected_adjacency(self):
        if self.uT is None:
            self.uT = to_symmetric(self.T, binary=False)
        return self.uT
    
    def set_nodes(self, nodes):
        self.sorted_nodes = nodes
        
    def get_nodes(self):
        return self.sorted_nodes
    
    def get_navitype(self):
        return self.navitype if self.navitype else ALLNAVITYPE
    
    def get_cs_filename(self, path, ext):
        return CS_FN.replace('<ONTO>',self.name).replace('<YEAR>',self.year).replace('<NAVITYPE>',self.get_navitype()).replace('<EXT>',ext)
        
        
    ################################################
    # Methods
    ################################################
    
    def load_clickstream_and_validate(self, cs_df, nodes, min_session_length=MIN_SESSION_LENGTH):
        df = cs_df.query("_ontology == @self.name and _year == @self.year")
        self._convert_DataFrame_to_DiGraph(df,nodes,min_session_length)
        self.sorted_nodes = sorted(list(self.H.nodes()))
        
    def create_adjacency_matrix(self, sorted_nodes=None):
        if self.H is None:
            raise ValueError("Clickstream graph has not been loaded!")
            return        
        if self.sorted_nodes is None and sorted_nodes is None:
            raise ValueError("Nodes have no order!")
            return
        self.T = nx.to_scipy_sparse_matrix(self.H, nodelist=self.sorted_nodes if sorted_nodes is None else sorted_nodes)
        
    def _convert_DataFrame_to_DiGraph(self, df, nodes, min_session_length=MIN_SESSION_LENGTH):
        
        edges = defaultdict(lambda:0)
        
        try:
            
            for name,group in df.groupby(['ip','_sessionid']):
                if len(group) < min_session_length:
                    continue

                dyad0=None
                dyad1=None
                seq0=None
                seq1=None

                for i,row in group.iterrows():

                    if dyad0 is None:
                        dyad0 = row._concept
                        seq0 = row._sequence
                        continue

                    if dyad1 is None:
                        dyad1 = row._concept
                        seq1 = row._sequence

                        if seq1 == (seq0+1) and dyad0 != dyad1:
                            if self.navitype is None or (self.navitype == row._navitype):
                                edges[(dyad0,dyad1)] += 1

                        dyad0 = dyad1
                        seq0 = seq1
                        dyad1 = None
                        seq1 = None

        except Exception as ex:
            printf(ex)
            printf('ERROR converting dataframe to digraph')
            return
        
        tmp = nx.DiGraph()
        tmp.add_weighted_edges_from([(e[0],e[1],w) for e,w in edges.items()])
        self.H = tmp.subgraph(nodes).copy()
        del(edges)
        
        printf('{}-{}-{}: {} concepts found, but {} kept (cros-val)'.format(self.name,self.year,self.navitype,tmp.number_of_nodes(),self.H.number_of_nodes()))
        del(tmp)

    ################################################
    # I/O
    ################################################
    def load_graph(self, path):
        self.H = read_graph(path, self.get_cs_filename(path,GRAPH_EXT))      
    
    def save_graph(self, path):
        save_graph(self.H, path, self.get_cs_filename(path,GRAPH_EXT))

    def save_adjacency(self, path):
        comment = 'Clickstreams\nOntology: {}\nYear: {}\nNavitype:{}'.format(self.name, self.year, self.get_navitype())
        field = 'integer'
        save_sparse_matrix(self.T, path, self.get_cs_filename(path,ADJ_EXT), comment=comment, field=field)
    
    def save_nodes(self, path):
        save_series(pd.Series(self.sorted_nodes), path, self.get_cs_filename(path,CSV_EXT))
        
    def load_adjacency(self, path):
        comment = 'Clickstreams\nOntology: {}\nYear: {}\nNavitype:{}'.format(self.name, self.year, self.get_navitype())
        field = 'integer'
        self.T = read_sparse_matrix(path, self.get_cs_filename(path,ADJ_EXT))
        
        
