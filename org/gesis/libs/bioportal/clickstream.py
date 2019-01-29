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
from org.gesis.libs.utils import save_graph
from org.gesis.libs.utils import save_adjacency
from org.gesis.libs.utils import save_series
from org.gesis.libs.bioportal.ontology import get_short_concept_name

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
CS_EXT = '.csv.bz2'
COMPRESSION = 'bz2'
CS_FN = '<ONTO>_<YEAR>.<EXT>'
GRAPH_EXT = 'gpickle'
ADJ_EXT = 'mtx'
CSV_EXT = 'csv'
CS_FN_SOURCE = 'BP_webpage_requests_<YEAR>.csv.bz2'
TMPFOLDER = '/bigdata/lespin/tmp/'

BREAK = 30

DIRECT_CLICK = 'DC'
DETAILS = 'DE'
EXPAND = 'EX'
DIRECT_URL = 'DU'
LOCAL_SEARCH = 'LS'
EXTERNAL_SEARCH = 'ES'
EXTERNAL_LINK = 'EL'
HOME_SEARCH = 'HS'
OTHERS = 'O'
NAVITYPES = [DIRECT_CLICK, DETAILS, EXPAND, DIRECT_URL, LOCAL_SEARCH, EXTERNAL_SEARCH, EXTERNAL_LINK, HOME_SEARCH]

BIOPORTAL_SUBDOMAIN = 'bioportal'
BIOPORTAL_DOMAIN = 'bioontology'
SEARCH_ENGINES = ['google','bing','yahoo','baidu','aol','ask','duckduckgo','dogpile','excite','wolframalpha','yandex','lycos','chacha']

key1 = '/ontologies/'
key2 = '/ajax_concepts/'
key3 = 'callback=children'
key4 = 'callback=load'
key5 = '/sty/'

########################################################################################
# Functions
########################################################################################     
def load_clickstream(path, year):
    try:
        fn = os.path.join(path,CS_FN_SOURCE.replace('<YEAR>',year))
        df = read_csv(fn,index_col=None,compression=COMPRESSION)
        printf('{} loaded!'.format(fn))
        return df
    except Exception as ex:
        printf(ex)
        printf('ERROR: CS{} NOT loaded!'.format(year))

def preprocess_clickstream(cs_df):
    cs_df.loc[:,'_ontology'] = None
    cs_df.loc[:,'_concept'] = None
    cs_df.loc[:,'_navitype'] = None
    cs_df.loc[:,'_request'] = cs_df.request.apply(lambda x: urllib.parse.unquote(x))
    cs_df.loc[:,'_ontology'] = cs_df._request.apply(lambda x: _get_ontology(x))
    cs_df.loc[:,'_concept'] = cs_df._request.apply(lambda x: _get_concept(x))  
    cs_df.loc[:,'_navitype'] = cs_df.apply(lambda row: _get_navitype(row._request, row.referer), axis=1)
    return cs_df

def preprocess_sessions(cs_df, lag_break_in_minutes=BREAK):
    cs_df.timestamp = pd.to_datetime(cs_df.timestamp)
    cs_df.sort_values(['ip','timestamp'], inplace=True)
    gt_break = cs_df.timestamp.diff() > datetime.timedelta(minutes=lag_break_in_minutes)
    diff_user = cs_df.ip != cs_df.ip.shift()
    session_id = (diff_user | gt_break).cumsum()
    cs_df['_sessionid'] = session_id
    cs_df.loc[:,'_year'] = pd.DatetimeIndex(cs_df.timestamp).year.unique()[0]
    return cs_df

def clean_clickstream(cs_df):
    nan = np.nan
    df = cs_df.query("statuscode == 200 & _ontology.notnull() & _concept.notnull() & _ontology != @nan & _concept != @nan").reset_index(drop=True)
    df.rename(columns={'Unnamed: 0':'original_id'}, inplace=True)
    df['_sequence'] = np.arange(1,len(df)+1,1) # to define transitions
    df = df.query("_navitype != 'O'")
    return df

def _get_ontology(request):
    '''
    @TODO:
    Consider this type of request:
    /ontologies/ICD9CM/classes/176
    Check first if it is necessary, or whether it is a triggered action
    '''    
    return urllib.parse.urljoin(request, urllib.parse.urlparse(request).path).replace(key1,'').replace(key2,'').replace('/','').upper()

def _get_concept(request):
    '''
    @TODO:
    Consider this type of request:
    /ontologies/ICD9CM/classes/176
    Check first if it is necessary, or whether it is a triggered action
    '''
    params = urllib.parse.parse_qs(urllib.parse.urlparse(request).query)
    concept = params['conceptid'][0].lower() if 'conceptid' in params else np.nan
    sty = concept is not np.nan and key5 in concept
    concept = urllib.parse.urljoin(concept, urllib.parse.urlparse(concept).path).split('/')[-1] if concept is not np.nan else concept 
    if sty:
        concept = '{}{}'.format(key5[1:],concept)
    return concept

def _get_navitype(request, referer):
    params = urllib.parse.parse_qs(urllib.parse.urlparse(request).query)
    try:
        referer_domain = tldextract.extract(referer).domain
        referer_subdomain = tldextract.extract(referer).subdomain
    except:
        referer_domain = ''
        referer_subdomain = ''
    
    if 'p' in params and params['p'][0] == 'classes' and 'conceptid' in params:
        
        if referer_subdomain == BIOPORTAL_SUBDOMAIN:
            
            if 'jump_to_nav' in params and params['jump_to_nav'][0] == 'true':
                return LOCAL_SEARCH
            
            elif '/search' in referer and '&query' in referer:
                return HOME_SEARCH
            
            elif '/ontologies' in referer and 'p=' in referer and 'conceptid=' in referer:
                return DETAILS
            
            else:
                return OTHERS
            
        elif referer_domain in SEARCH_ENGINES:
            return EXTERNAL_SEARCH
        
        elif referer_domain in ['','-',np.nan,'NaN','nan',None]:
            return DIRECT_URL
        
        else:
            return EXTERNAL_LINK

    elif 'callback' in params and 'conceptid' in params and referer_subdomain == BIOPORTAL_SUBDOMAIN:
        
        if params['callback'][0] == 'load':
            return DIRECT_CLICK
        
        elif params['callback'][0] == 'children':
            return EXPAND

    return OTHERS
        

def main():
    printf('class clickstreams')
          
if __name__ == '__main__':
    main()