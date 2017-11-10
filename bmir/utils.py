__author__ = 'espin'

########################################################################
# IMPORTS
########################################################################
import os
import sys
import ast
import json
import math
import gzip
import pickle
import operator
import traceback
import pycountry
import tldextract
import numpy as np
import nltk, string
import pandas as pd
from scipy import io
import networkx as nx
from scipy import stats
from datetime import datetime
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
import seaborn as sns;
#sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})

#nltk.load('/home/lespin/nltk_data/tokenizers/punkt/english.pickle')

########################################################################
# CONSTANTS
########################################################################
FIELD_SEP = ','
COMPRESSION = 'bz2'
TSFORMAT = '%Y-%m-%d %H:%M:%S'
LOGFILE = None


####################################################################
### TEXT
####################################################################
###http://stackoverflow.com/questions/8897593/similarity-between-two-text-documents

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def text_similarity(text1, text2):
    if len(text1.strip().replace(' ','')) == len(text2.strip().replace(' ','')) and len(text1.strip().replace(' ','')) == 0:
        return 1.
    else:
        vector = [text1, text2]
        try:
            tfidf = vectorizer.fit_transform(vector)
            return ((tfidf * tfidf.T).A)[0,1]
        except Exception as ex:
            printf('error: {}'.format(ex))
            printf('text1: {} ({})'.format(text1,type(text1)))
            printf('text2: {} ({})'.format(text2,type(text2)))
            printf('vector: {}'.format(vector))
        return 0.

########################################################################
# SORT
########################################################################

def sort_dict(data,by_value=False,desc=False):
    return sorted(data.items(), key=operator.itemgetter(by_value), reverse=desc)

########################################################################
# FUNCTIONS I/O
########################################################################
def validateParameters(arguments):
    argv = sys.argv
    if len(argv)-1 != len(arguments):
        printf('Usage:')
        printf('python {} {}'.format(argv[0],' '.join(arguments)))
        sys.exit(0)
    else:
        printf('Arguments: \n{}'.format('\n'.join(['{}:{}'.format(i,a) for i,a in enumerate(argv)])))

def getParameter(index):
    val = None
    try:
        val = sys.argv[index]
        tmp = str(val).lower()

        if tmp in ['true','false']:
            return (tmp == 'true')

        if tmp == 'none':
            return None

        try:
            val = ast.literal_eval(val)
        except:
            val = val

    except:
        printf('Index {} does not exist'.format(index))
        sys.exit(0)

    return val

def set_log_file(output,fn):
    global LOGFILE
    LOGFILE = os.path.join(output,fn)

def printf(txt):
    strtowrite = "[{}] {}".format(datetime.now(), txt)
    print(strtowrite)
    write_log(strtowrite)

def write_log(txt):
    if LOGFILE is not None:
        f = open(LOGFILE,'a')
        f.write(txt)
        f.write('\n')
        f.close()

def stats_summary(data):
    printf('=== Statistics ===')
    printf('min: {}'.format(min(data)))
    printf('max: {}'.format(max(data)))
    printf('mean: {}'.format(np.mean(data)))
    printf('mode: {}'.format(stats.mode(data)))

########################################################################
# FILES
########################################################################

########################################
# LOAD / READ
########################################
def load(fn,**kwargs):
    file_extension = _get_extension(fn)
    if file_extension == '.p':
        obj = _load_pickle(fn)
    if file_extension == '.mtx':
        obj = _load_sparse_matrix(fn)
    if file_extension == '.gpickle':
        obj = _load_gpickle(fn)
    if file_extension == '.json':
        obj = _load_json(fn)
    if file_extension == '.pklz':
        obj = _load_compressed_pickle(fn)
    if file_extension == '.csv':
        obj = _load_csv(fn,**kwargs)

    if obj is not None:
        printf('{} loaded!'.format(fn))
    else:
        printf('ERROR: {} NOT loaded.'.format(fn))
    return obj

def _load_pickle(fn):
    try:
        with open(fn,'r') as f:
            obj = pickle.load(f)
        return obj
    except Exception as ex:
        traceback.print_exc()
        return None

def _load_sparse_matrix(fn):
    try:
        obj = io.mmread(fn)
        return obj.tocsr()
    except Exception as ex:
        traceback.print_exc()
        return None

def _load_gpickle(fn):
    try:
        return nx.read_gpickle(fn)
    except Exception as ex:
        traceback.print_exc()
        return None

def _load_json(fn):
    with open(fn,'r') as f:
        obj = json.load(f)
    return obj

def _load_compressed_pickle(fn):
    try:
        with gzip.open(fn,'r') as f:
            obj = pickle.load(f)
        return obj
    except Exception as ex:
        traceback.print_exc()
        return None

def _load_csv(fn,**kwargs):
    try:
        if 'converters' in kwargs:
            obj = pd.read_csv(fn,sep=FIELD_SEP, converters=kwargs['converters'])
        else:
            obj = pd.read_csv(fn,sep=FIELD_SEP)
        return obj
    except Exception as ex:
        traceback.print_exc()
        return None


########################################
# SAVE / WRITE
########################################
def save(fn,obj):
    file_extension = _get_extension(fn)
    if file_extension == '.p':
        flag = _save_pickle(fn,obj)
    if file_extension == '.mtx':
        flag = _save_sparse_matrix(fn, obj)
    if file_extension == '.gpickle':
        flag = _save_gpickle(fn,obj)
    if file_extension == '.json':
        flag = _save_json(fn,obj)
    if file_extension == '.pklz':
        flag = _save_compressed_pickled(fn,obj)
    if file_extension == '.csv':
        flag = _save_csv(fn,obj)

    if flag:
        printf('{} saved!'.format(fn))
    else:
        printf('ERROR: {} NOT saved.'.format(fn))
    return flag

def _save_pickle(fn,obj):
    try:
        with open(fn,'w') as f:
            pickle.dump(obj,f)
    except:
        return False
    return True

def _save_sparse_matrix(fn, obj):
    try:
        io.mmwrite(fn, obj)
    except:
        return False
    return True

def _save_gpickle(fn,obj):
    try:
        nx.write_gpickle(obj,fn)
    except:
        return False
    return True

def _save_json(fn,obj):
    try:
        with open(fn,'w') as f:
            json.dump(obj,f)
    except:
        return False
    return  True

def _save_compressed_pickled(fn,obj):
    try:
        with gzip.open(fn,'w') as f:
            pickle.dump(obj,f)
    except:
        return False
    return  True

def _save_csv(fn,obj):
    try:
        if type(obj) == pd.DataFrame:
            obj.to_csv(fn, sep=FIELD_SEP)
    except:
        return False
    return  True

########################################
# PATHS
########################################
def exists(fn):
    return os.path.exists(fn)

def _get_extension(fn):
    filename, file_extension = os.path.splitext(fn)
    return file_extension

def getFullPath(path,*args):
    for fn in args:
        path = os.path.join(path,fn)
    return path

def getFileName(path):
    return os.path.basename(path)

def createFolder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except:
        #traceback.print_exc()
        printf('{} already exists.'.format(path))
    return path


########################################################################
# DATAFRAMES
########################################################################
def dataframe_read(fn,sortbycolumns=None,ontology=None):
    df = pd.read_csv(fn,sep=FIELD_SEP,compression=COMPRESSION)
    if 'timestamp' in df:
        df['timestamp'] = pd.to_datetime(df.timestamp)
    if ontology is not None:
        df = df.loc[df['ontology'] == ontology]
    if sortbycolumns is not None:
        df = dataframe_sortby(df,sortbycolumns)
    return df

def dataframe_sortby(df,columns):
    return df.sort_values(by=columns)

def dataframe_groupby(df,columns):
    return df.groupby(columns, as_index=False)

def _as_string(x):
    return '%s' % x

def _as_int(x):
    return '%d' % x

def _float_as_string(x):
    return '%1.2f' % x

def _string_as_float(x):
    return float(x)

########################################################################
# TIME FUNCTIONS
########################################################################

def delta_time(scale,timestamp1,timestamp2):
    if scale == 'seconds':
        return _delta_seconds(timestamp1,timestamp2)
    if scale == 'minutes':
        return _delta_minutes(timestamp1,timestamp2)

def _delta_seconds(timestamp1,timestamp2,tsformat=TSFORMAT):
    return _delta_time(timestamp1,timestamp2,tsformat).total_seconds()

def _delta_minutes(timestamp1,timestamp2,tsformat=TSFORMAT):
    s = _delta_seconds(timestamp1,timestamp2,tsformat)
    m = s / 60.
    #if m > 1: printf('{} secs --> {} min'.format(s,m))
    return m

def _delta_time(timestamp1,timestamp2,tsformat=TSFORMAT):
    if isinstance(timestamp1, basestring) and isinstance(timestamp2, basestring):
        t1 = datetime.strptime(timestamp1,tsformat)
        t2 = datetime.strptime(timestamp2,tsformat)
    else:
        t1 = timestamp1
        t2 = timestamp2
    d = abs(t1 - t2)
    return d

########################################################################
# NETWORK
########################################################################

def get_max_hops(G):
    printf('Calculating max HOP...')
    paths = nx.all_pairs_shortest_path(G)
    ### -2 because we remove first and last node from the list of shortest paths
    ### those are the source and target nodes
    ### ex: {1: {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 4], 5: [1, 2, 5]} }
    ### So, (1) to (5) goes: (1) -> (2) -> (5) therefore, 3 nodes in the path minus 2, 1 HOP
    return max([len(hop)-2 for s,obj in paths.items() for t,hop in obj.items() if s != t])

def get_average_degree(G, weight=None):
    if G.number_of_nodes() == 0 :
        return 0
    ### unique edges
    if weight is None:
        if nx.is_directed(G):
            return G.number_of_edges() / float(G.number_of_nodes()) ### weighted
        return 2 * G.number_of_edges() / float(G.number_of_nodes()) ### unweighted

    ### multiedges
    if nx.is_directed(G):
        return G.size(weight=weight) / float(G.number_of_nodes()) ### weighted
    return 2 * G.size(weight=weight) / float(G.number_of_nodes()) ### unweighted

def get_score_overlap(structure_adjacency, intersection_adjacency, weighted=False):

    avg_neighbors = structure_adjacency.sum() / float(structure_adjacency.shape[0])

    if avg_neighbors == 0:
        return 0

    if weighted:
        return intersection_adjacency.sum() / float(avg_neighbors)
    return intersection_adjacency.nnz / float(avg_neighbors)


def get_score_overlap_old(graph, interception, weight=None):

    avg_neighbors = 0 if graph.number_of_nodes() == 0 else sum([len(graph[n]) for n in graph]) / float(graph.number_of_nodes())

    if avg_neighbors == 0:
        return 0

    ### unique edges
    if weight is None:
        if nx.is_directed(interception):
            return interception.number_of_edges() / float( avg_neighbors )
        return 2 * interception.number_of_edges() / float( avg_neighbors )

    ### multiedges
    if nx.is_directed(interception):
        return interception.size(weight=weight) / float( avg_neighbors )
    return 2 * interception.size(weight=weight) / float( avg_neighbors )

def convert_directed_to_undirected(graph,weighted=False):
    H = nx.Graph()
    H.add_nodes_from(graph.nodes())
    if weighted:
        H.add_edges_from(graph.edges_iter(), weight=0)
        for u, v, d in graph.edges_iter(data=True):
            H[u][v]['weight'] += d['weight']
    else:
        H.add_edges_from(graph.edges_iter())
    return H

def get_adjacency_matrix(G,nodelist=None,weighted=False):
    # return nx.adjacency_matrix(G,nodelist=nodelist,weight='weight' if weighted else None)
    return nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight='weight' if weighted else None, format='csr')

def combine_directed_graphs(G,H):
    tmp = nx.DiGraph()

    if G is not None:
        tmp.add_nodes_from(G.nodes())
        tmp.add_edges_from(G.edges_iter(), weight=0)
        for u, v, d in G.edges_iter(data=True):
            tmp[u][v]['weight'] += d['weight']

    if H is not None:
        tmp.add_nodes_from(H.nodes())

        for u, v, d in H.edges_iter(data=True):
            if not tmp.has_edge(u,v):
                tmp.add_edge(u,v,weight=0)
            tmp[u][v]['weight'] += d['weight']

    return tmp

def create_directed_graph(nodes, edges, weight):
    H = nx.DiGraph()
    if nodes is not None:
        H.add_nodes_from(nodes)
    H.add_edges_from(edges,weight=weight)
    return H

def create_undirected_graph(nodes, edges):
    H = nx.Graph()
    if nodes is not None:
        H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    return H

def add_weights_to_edges(H, G):
    for u, v, d in H.edges_iter(data=True):
        H[u][v]['weight'] += G[u][v]['weight']
    return H

def are_siblings(u,v,G):
    s1 = set(G.predecessors(u))
    s2 = set(G.predecessors(v))
    return len(s1.intersection(s2)) > 0

def add_attribute_nodes(key,data,G):
    if data is not None:
        for node, obj in data.items():
            if node in G:
                G.node[node][key] = obj
            else:
                printf('{} not in graph'.format(node))
    printf('Attributes {} added to the graph!'.format(key))
    return G

########################################################################
# PLOTS
########################################################################

def _plot(fig,ax,fn,x,y,**kwargs):
    if 'bcolor' in kwargs:
        ax.patch.set_facecolor(kwargs['bcolor'])
    if 'logy' in kwargs:
        #ax.set_yscale('log')
        plt.yscale('log', nonposy='clip')
    if 'logx' in kwargs:
        #ax.set_xscale('log')
        plt.xscale('log', nonposy='clip')
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=10)
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=10)
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if 'xticks' in kwargs:
        tmp = kwargs['xticks']
        plt.xticks(x,tmp['ticks'],rotation=tmp['rotation'],fontsize=tmp['fontsize'])
    if 'yticks' in kwargs:
        tmp = kwargs['yticks']
        plt.yticks(y,tmp['ticks'],rotation=tmp['rotation'],fontsize=tmp['fontsize'])
    if 'text' in kwargs:
        tmp = kwargs['text']
        ax.text(tmp['x'], tmp['y'], tmp['text'], fontsize=tmp['fontsize'], bbox=tmp['bbox'])
    if 'grid' in kwargs:
        plt.grid(kwargs['grid']) # True or False
    if 'locatorx' in kwargs:

        for i,label in enumerate(ax.xaxis.get_ticklabels()):
            label.set_visible(False)

        bins = int(round((len(ax.xaxis.get_ticklabels())+1) / float(kwargs['locatorx'])))
        if bins == 0:
            bins = 1

        for label in ax.xaxis.get_ticklabels()[::bins]:
            label.set_visible(True)

    if 'locatory' in kwargs:
        for i,label in enumerate(ax.yaxis.get_ticklabels()):
            label.set_visible(False)

        bins = int(round((len(ax.yaxis.get_ticklabels())+1) / float(kwargs['locatory'])))
        if bins == 0:
            bins = 1

        for label in ax.yaxis.get_ticklabels()[::bins]:
            label.set_visible(True)

    try:
        plt.tight_layout()
    except Exception as e:
        printf('WARNING: _plot(), {}'.format(e.message))

    plt.savefig(fn, facecolor=fig.get_facecolor())
    plt.close()

def plot_line(x,y,fn,**kwargs):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x, y)
    _plot(fig,ax,fn,x,y,**kwargs)

def plot_bars(x,y,fn,**kwargs):
    fig = plt.figure()
    ax = plt.gca()
    width = 0.35
    ax.bar(x, y, width)
    _plot(fig,ax,fn,x,y,**kwargs)

def plot_matrix(m,fn,**kwargs):
    grid_kws = {"height_ratios": (.95, .05), "hspace": .4 if 'hspace' not in kwargs else kwargs['hspace']}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(m if isinstance(m,np.ndarray) or isinstance(m,pd.DataFrame) else m.toarray(), ax=ax,
        annot=False if 'annotation' not in kwargs else kwargs['annotation']['annot'],
        annot_kws=None if 'annotation' not in kwargs else kwargs['annotation']['kws'],
        cbar_ax=cbar_ax,
        fmt=".2f",
        cbar_kws={"orientation": "horizontal"},
        yticklabels=kwargs['yticks']['ticks'],
        xticklabels=kwargs['xticks']['ticks'])

    if 'title' in kwargs:
        sns.plt.suptitle(kwargs['title'])

    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')

    if 'xticks' in kwargs:
        tmp = kwargs['xticks']
        plt.setp( ax.xaxis.get_majorticklabels(),  rotation=tmp['rotation'], horizontalalignment='center', fontsize=tmp['fontsize'] )

    if 'yticks' in kwargs:
        tmp = kwargs['yticks']
        plt.setp( ax.yaxis.get_majorticklabels(),  rotation=tmp['rotation'], horizontalalignment='right', fontsize=tmp['fontsize'] )


    cbar_ax.set_xlabel(kwargs['bartitle'], labelpad=10)
    cbar_ax.xaxis.set_label_position('bottom')

    plt.gca().invert_yaxis()

    plt.savefig(fn, dpi=1200, bbox_inches='tight')
    plt.close()

    printf('matrix plot saved: {}'.format(fn))

def clustermap(df, fn, **kwargs):
    try:
        g = sns.clustermap(df,**kwargs)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.savefig(fn)
    except Exception as e:
        printf('ERROR clustermap: {}'.format(e.message))
        return None

    printf('clustermap saved: {}'.format(fn))
    return g

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fn = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if fn is not None:
        plt.tight_layout()
        plt.savefig(fn)
    else:
        plt.show()
    plt.close()

########################################################################
# URLs
########################################################################

def get_domain(url):
    # try:
    #     parsed_uri = urlparse( url )
    #     domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
    # except Exception as e:
    #     #traceback.print_exc()
    #     printf('url:{} | error:{}'.format(url,e.message))
    #     return None

    try:
        tmp = tldextract.extract(url)
        domain = tmp.registered_domain

        if domain == '':
            domain = None

    except Exception as e:
        if url is not None:
            printf('get_domain: url:{} | error:{}'.format(url,e.message))

            if url in ['','-']:
                return ''

        return None

    # if domain == ':///' or domain == '':
    #     printf('\n\n------>>> url:{}\n\n'.format(url))

    return  domain

def get_domain_name(url):
    try:
        tmp = tldextract.extract(url)
        d = tmp.domain
    except Exception as ex:
        if url is not None:
            printf('get_domain_name: url:{} | error:{}'.format(url,ex.message))
        return None
    return d

def get_country(url):
    #ExtractResult(subdomain='forums.news', domain='cnn', suffix='com')
    try:
        tmp = tldextract.extract(url)
        tmp = tmp.suffix.split('.')[0]
        tmp = 'gb' if tmp == 'uk' else tmp
        c = pycountry.countries.lookup(tmp)
    except Exception as ex:
        printf('url:{} | error:{}'.format(url,ex.message))
        c = None
    return c


########################################################################
# OTHERS
########################################################################

def get_entropy(sequence):
    return entropy( [sequence.count(v)/float(len(sequence)) for v in set(sequence)] , base=2)

    # freq = {item:sequence.count(item) for item in set(sequence)}
    # total = float(len(sequence))
    # prob = {i:c/total for i,c in freq.items()}
    # value = sum([p * np.log2(p) for i,p in prob.items()])
    # #printf('Max entropy log2(k) = {} | k = {}'.format(np.log2(len(freq.items())), len(freq.items())))
    # return 0 if abs(value) == 0. else value * -1