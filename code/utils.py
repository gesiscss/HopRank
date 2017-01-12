__author__ = 'espin'

#############################################################################
# Dependences
#############################################################################
import sys
from scipy import io
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})
import os
import networkx as nx
import ast

#############################################################################
# Functions
#############################################################################
def getParameter(index):
    val = None
    try:
        val = sys.argv[index]
        val = str(val).lower()

        if val in ['true','false']:
            val = (val == 'true')

        try:
            val = ast.literal_eval(val)
        except:
            val = val

    except:
        print('Index {} does not exist'.format(index))

    return val

def exists(fn):
    return os.path.exists(fn)

def saveSparseMatrix(A,fn):
    io.mmwrite(fn, A)

def loadSparseMatrix(fn):
    return io.mmread(fn)

def saveAdjacencyMatrix(G,fn):
    try:
        A = nx.adjacency_matrix(G)
        saveSparseMatrix(A,fn)
        print('Adjacency matrix for {} saved!'.format(G.graph['name']))
    except:
        print('ERROR: Adjacency matrix for {} not saved.'.format(G.graph['name']))

def saveGraph(G,fn):
    try:
        nx.write_gpickle(G, fn)
        print('Graph {} saved!'.format(G.graph['name']))
    except:
        print('ERROR: Graph {} not saved.'.format(G.graph['name']))

def loadGraph(fn):
    G = None
    try:
        G = nx.read_gpickle(fn)
        print('Graph {} loaded!'.format(G.graph['name']))
    except:
        print('ERROR: Graph not loaded: {}'.format(fn))
    return G

def plot_matrix(m,fn,**kwargs):
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(m.toarray(), ax=ax,
        # annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"})
    ax.set_xlabel(kwargs['xtitle'])
    ax.set_ylabel(kwargs['ytitle'])
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, horizontalalignment='center', fontsize=4 )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=0, horizontalalignment='center', x=1.0, fontsize=4 )

    cbar_ax.set_title(kwargs['bartitle'])

    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot matrix done!')
    plt.close()

