__author__ = "Lisette Espin-Noboa"
__copyright__ = "Copyright 2018, HopRank"
__credits__ = ["Florian Lemmerich", "Markus Strohmaier", "Simon Walk", "Mark Musen"]
__license__ = "GPL"
__version__ = "1.0.3"
__maintainer__ = "Lisette Espin-Noboa"
__email__ = "Lisette.Espin@gesis.org"
__status__ = "Developing"

########################################################################################
# Dependencies
########################################################################################
import os
import time
import datetime
import numpy as np
import pandas as pd
import networkx as nx
# from scipy.io import mmwrite
# from scipy.io import mmread
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz

########################################################################################
# Functions
########################################################################################

def printf(txt):
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print('{}\t{}'.format(ts,txt))
    
def read_csv(fn,index_col=None,compression=None):
    if not os.path.exists(fn):
        raise ValueError("{} does not exist!".format(fn))
        return    
    return pd.read_csv(fn,index_col=None, compression=compression, encoding = "ISO-8859-1", low_memory=False)

def log(path, prefix, cap):
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H')
    fn = os.path.join(path,'{}_{}.log'.format(prefix,ts))
    try:
        with open(fn, 'a') as f:
            f.write(cap.stdout)
        printf('{} saved!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT saved!'.format(fn))

########################################################################################
# Functions: pandas
########################################################################################

def read_series(path, fn):    
    s = None
    try:
        fn = os.path.join(path,fn)
        s = pd.read_csv(fn, squeeze=True)
        printf('{} loaded!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT loaded!'.format(fn)) 
    return s 
        
def save_series(s, path, fn):
    if s is None:
        raise ValueError("Series has not been loaded!")
        return 
    try:
        fn = os.path.join(path,fn)
        s.to_csv(fn, index=True)
        printf('{} saved!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT saved!'.format(fn))    

########################################################################################
# Functions: Graphs
########################################################################################

def read_graph(path, fn):
    G = None
    try:
        fn = os.path.join(path,fn)
        G = nx.read_gpickle(fn)
        printf('{} loaded!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT loaded!'.format(fn))
    return G
        
def save_graph(G, path, fn):
    if G is None:
        raise ValueError("Graph has not been loaded!")
        return 
    try:
        fn = os.path.join(path,fn)
        nx.write_gpickle(G, fn)
        printf('{} saved!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT saved!'.format(fn))

def weighted_to_undirected(G):
    tmp = nx.Graph()
    tmp.add_edges_from(G.edges(), weight=0)
    for u, v, d in G.edges(data=True):
        tmp[u][v]['weight'] += d['weight']
    return tmp

########################################################################################
# Functions: Matrices
########################################################################################            
def save_sparse_matrix(A, path, fn, comment=None, field=None):
    if A is None:
        raise ValueError("Sparse matrix has not been loaded!")
        return 
    try:
        fn = os.path.join(path,fn)
        save_npz(fn, A.tocsr(), True)
        printf('{} saved!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT saved!'.format(fn))        

def read_sparse_matrix(path, fn): 
    A = None
    try:
        fn = os.path.join(path,fn)
        A = load_npz(fn).tocsr()
        printf('{} loaded!'.format(fn))
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT loaded!'.format(fn))         
    return A

def to_symmetric(sparse_matrix,binary=True):
    sparse_matrix=lil_matrix(sparse_matrix)
    rows, cols = sparse_matrix.nonzero()
    if binary:
        sparse_matrix[cols, rows] = sparse_matrix[rows, cols]
    else:
        sparse_matrix[cols, rows] += sparse_matrix[rows, cols]
    return sparse_matrix.tocsr()

def get_khop_with_partial_results(M, maxk):
    previous = None
    
    for k in range(1,maxk+1,1):

        if k == 1:            
            hop = M.copy()
        else:
            printf('accumulating previous hops...')
            if previous is None:
                previous = hop.copy()                
            else:
                previous += hop.copy()
            printf('multiplying: product.dot(m)...')
            hop = hop.dot(M)
        
        hop = (hop>0).tocsr().astype(np.int8)
        
        if previous is not None:
            printf('substracting previous hops from {}hop...'.format(k))
            _hop = hop - previous
        else:
            _hop = hop.copy()

        printf('>0...') 
        _hop = (_hop>0)
        printf('eliminating 0s...')
        _hop.eliminate_zeros()
        #_hop = _hop.tolil()
        printf('setting diagonal to zero...')
        _hop.setdiag(0)
        printf('eliminating 0s...')
        _hop.eliminate_zeros()
        printf('to csr int...')
        _hop = _hop.tocsr().astype(np.int8)  
        printf('done {}-hop!'.format(k))
        yield k,_hop
        
        
