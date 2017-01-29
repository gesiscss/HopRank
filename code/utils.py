__author__ = 'espin'

#############################################################################
# Local Dependences
#############################################################################

#############################################################################
# System Dependences
#############################################################################
import sys
from scipy import io
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})
import os
import networkx as nx
import ast
from datetime import datetime
import pickle
from collections import Counter
import numpy as np
import time
import operator
import itertools

#############################################################################
# Functions
#############################################################################
def getParameter(index):
    val = None
    try:
        val = sys.argv[index]
        tmp = str(val).lower()

        if tmp in ['true','false']:
            return (val == 'true')

        if tmp == 'none':
            return None

        try:
            val = ast.literal_eval(val)
        except:
            val = val

    except:
        print('Index {} does not exist'.format(index))

    return val

#############################################################################
# CONSTANTS
#############################################################################
HOP1 = 1
HOP2 = 2


##############################################################################
# Log
##############################################################################
class Utils(object):

    def __init__(self,logfile):
        self.logfile = logfile
        self.inittime = None

    ####################################################################
    ### LOGGING
    ####################################################################
    def loginit(self):
        self.inittime = time.time()
        self.log('INIT')

    def logend(self):
        self.log('END: {} secods.'.format(time.time() - self.inittime))

    def log(self,msg):
        strtowrite = "[{}] {}".format(datetime.now(), msg)
        print strtowrite
        self.writeText(strtowrite)

    ####################################################################
    ### TIME
    ####################################################################
    def deltaTimestamps(self, timestamp1, timestamp2, tsformat):
        if isinstance(timestamp1, basestring) and isinstance(timestamp2, basestring):
            t1 = datetime.strptime(timestamp1,tsformat)
            t2 = datetime.strptime(timestamp2,tsformat)
        else:
            t1 = timestamp1
            t2 = timestamp2
        d = abs(t1 - t2)
        #self.log('{} - {} = {}'.format(t1,t2,d.total_seconds()))
        return d

    ####################################################################
    ### FILES
    ####################################################################
    def savePickle(self,data,fn):
        try:
            pickle.dump( data, open( fn, "wb" ) )
        except:
            self.log('ERROR: Pickle not saved: {}'.format(fn))

    def loadPickle(self, fn):
        obj = None
        try:
            with open(fn,'rb') as f:
                obj = pickle.load(f)
        except:
            self.log('ERROR: Pickle not loaded: {}'.format(fn))
        return obj

    def writeText(self,msg):
        f = open(self.logfile,'a')
        f.write(msg)
        f.write('\n')
        f.close()

    def saveAdjacencyMatrix(self,G,fn):
        try:
            A = nx.adjacency_matrix(G,weight='weight')
            self.saveSparseMatrix(A,fn)
            self.log('Adjacency matrix for {} saved!'.format(G.graph['name']))
        except Exception as ex:
            self.log('ERROR: Adjacency matrix for {} not saved.'.format(G.graph['name']))
            self.log(ex.message)
            print(ex)

    def saveSparseMatrix(self,A,fn):
        io.mmwrite(fn, A)

    def saveGraph(self,G,fn):
        try:
            nx.write_gpickle(G, fn)
            self.log('Graph {} saved!'.format(G.graph['name']))
        except:
            self.log('ERROR: Graph {} not saved.'.format(G.graph['name']))

    def loadGraph(self,fn):
        G = None
        try:
            G = nx.read_gpickle(fn)
            self.log('Graph {} loaded!'.format(G.graph['name']))
        except:
            self.log('ERROR: Graph not loaded: {}'.format(fn))
        return G

    def loadSparseMatrix(self,fn):
        return io.mmread(fn)

    def exists(self,fn):
        return os.path.exists(fn)

    def getYearFromFileName(self, fn):
        ### format: BP_webpage_requests_2015.csv.bz2
        try:
            year = str(int(fn.split('.')[0].split('_')[-1])) #if not int then except
        except:
            year = ''
        return year

    def generateSimplePath(self, root, year, source):
        ### root: results path
        ### year: dataset year
        ### type: ontologies, clickstream, apirequests, transitions
        path = os.path.join(os.path.join(root,year),source)
        if self.createFolder(path):
            return path
        return None

    def generatePath(self, root, year, source, filterin, filterout):

        path = os.path.join(os.path.join(root,year),source)
        b = 'FilterOut'

        ### Filterout: Removing records
        if filterout is None:
            b = '{}_None'.format(b)
        else:
            for k,v in self.filterout.items():
                b = '{}_{}_{}'.format(b,k,v)
                b = b.replace(' ','_')

        path = os.path.join(path,b)
        if self.createFolder(path):
            return path
        return None

    def createFolder(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                return True
            return True
        except:
            return False

    ####################################################################
    ### PLOTS
    ####################################################################

    def plot_matrix(self,m,fn,**kwargs):
        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        ax = sns.heatmap(m if isinstance(m,np.ndarray) else m.toarray(), ax=ax,
            # annot=True,
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"})
        ax.set_xlabel(kwargs['xtitle'])
        ax.set_ylabel(kwargs['ytitle'])
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')

        plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, horizontalalignment='center', fontsize=8 )
        plt.setp( ax.yaxis.get_majorticklabels(), rotation=0, horizontalalignment='center', x=1.0, fontsize=4 )

        cbar_ax.set_title(kwargs['bartitle'])
        plt.gca().invert_yaxis()

        plt.savefig(fn, dpi=1200, bbox_inches='tight')

        self.log('- plot matrix done!')
        plt.close()

    def scatterPlot(self, data, fn, **kwargs):
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(data['x'] ,data['y'] , c='blue', alpha=0.05 if 'alpha' not in kwargs else kwargs['alpha'], edgecolors='none')
        if 'logy' in kwargs:
            ax.set_yscale('log')
        if 'logx' in kwargs:
            ax.set_xscale('log')
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'], fontsize=10)
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'], fontsize=10)
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

    def plotGroupBars(self, df,title,fn,**kwargs):

        labelkey = 'action' if 'labelkey' not in kwargs else kwargs['labelkey']
        groups = ['as_entry','as_exit'] if 'groups' not in kwargs else kwargs['groups']
        ylog = False if 'ylog' not in kwargs else kwargs['ylog']
        xticks = False if 'xticks' not in kwargs else kwargs['xticks']
        posdelta = False if 'posdelta' not in kwargs else kwargs['posdelta']
        xlabel = None if 'xlabel' not in kwargs else kwargs['xlabel']
        legend = kwargs['legend'] if 'legend' in kwargs else [s.replace('_', ' ').title() for s in groups]

        colors = ['#EE3224','#F78F1E','#F6FF33','#C1FF33','#33FF3F','#33FFF9','#33C4FF','#3339FF','#8A33FF','#FC33FF','#FF33BE','#900C3F','#000000']
        #red,orange,yellow,light green,green,cyan,blue,dark blue,purple,fucsia,pink,dar red,black

        # Setting the positions and width for the bars
        pos = list(range(len(df[labelkey])))
        width = 0.25

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(10,5))
        for i,row in df.iterrows():
            for j,g in enumerate(groups):
                plt.bar([p + ((width*j) if posdelta else 0.) for p in pos],
                    df[g],
                    width,
                    alpha=0.5,
                    color=colors[j],
                    label=row[labelkey])

        # Scale
        if ylog:
            ax.set_yscale("log", nonposy='clip')

        # Set the x axis label
        if xlabel:
            ax.set_xlabel(xlabel)

        # Set the y axis label
        ax.set_ylabel('Frequency')

        # Set the chart's title
        ax.set_title(title)

        # Set the position of the x ticks
        ax.set_xticks([p + ( (width*len(groups)/2.) if posdelta else (width/2.)) for p in pos])

        # Set the labels for the x ticks
        if xticks:
            xticklabels = df[labelkey]
        else:
            xticklabels = list(pos)
        ax.set_xticklabels(xticklabels,rotation=90,size=10 if len(pos) < 20 else 6)

        # Setting the x-axis and y-axis limits
        plt.xlim(min(pos)-width, max(pos)+width*4)

        rows = df.shape[0]
        tmp = range(rows)
        for r in range(rows):
            tmp[r] = 0
            for g in groups:
                tmp[r] += df[g][r]

        plt.ylim([0.9 if ylog else 0., max(tmp)] )

        # Adding the legend and showing the plot
        plt.legend(legend, loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(fn)
        return

    ####################################################################
    ### MATHS
    ####################################################################
    def avg(self, lst):
        if len(lst) > 0:
            return sum(lst)/float(len(lst))
        return 0

    def median(self, lst):
        return np.median(np.array(lst))

    def mode(self, lst):
        ### http://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
        data = Counter(lst)
        return [kv[0] for kv in data.most_common() if kv[1] == data.most_common(1)[0][1]]
        #data.most_common()         # Returns all unique items and their counts
        #return data.most_common(1)  # Returns the highest occurring item

    ####################################################################
    ### DICTIONARIES
    ####################################################################
    def sortDict(self, x, byValue=False, descendent=False):
        return sorted(x.items(), key=operator.itemgetter(byValue), reverse=descendent)

    ####################################################################
    ### GRAPHS
    ####################################################################

    def areNodesSiblings(self, G, source, target):
        if source in G and target in G:
            ps = G.predecessors(source)
            pt = G.predecessors(target)
            flag = len(list(set(ps).intersection(pt))) > 0
            #self.log('{}->{}:{}'.format(s,t,flag))
            return flag
        return False

    def areNodesNeighbors(self,G,source,target):
        try:
            flag = target in nx.all_neighbors(G,source)
            return flag
        except:
            return False

    def areNodesConnected(self,G,source,target):
        return self.areNodesNeighbors(G,source,target) or self.areNodesSiblings(G,source,target)


    ########################################
    ### SEQUENTIAL
    ########################################

    def getConnectedNodes(self,G):
        try:
            paths = self.getNeighbors(G)
            self.log('# neighbors: {}'.format(len(paths)))

            siblings = self.getSiblings(G)
            self.log('# siblings: {}'.format(len(siblings)))

            return paths.union(siblings)
        except Exception as ex:
            self.log('ERROR: {}'.format(ex.message))
            return set()

    def getNeighbors(self, G):
        return set(tuple(sorted((source,target))) for source,neighbors in {node:nx.all_neighbors(G,node) for node in G.nodes()}.items() for target in neighbors)

    def getSiblings(self, G):
        return set(tuple(sorted(edge)) for siblings in [tuple(G.successors(node)) for node in G.nodes()] if len(siblings) > 1 for edge in itertools.combinations(siblings, 2))

    ####################################################################
    ### OTHERS
    ####################################################################

    ###http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]




####################################################################
### DEPRECATED
####################################################################


# def _getNeighbors(G, nodes):
#     return set(tuple(sorted((source,target))) for source,neighbors in {node:nx.all_neighbors(G,node) for node in nodes}.items() for target in neighbors)
#
# def _getSiblings(G, nodes):
#     return set(tuple(sorted(edge)) for siblings in [tuple(G.successors(node)) for node in nodes] if len(siblings) > 1 for edge in itertools.combinations(siblings, 2))
#
# def _getConnectedEdges(G, nodes):
#     neighbors = set(tuple(sorted((source,target))) for source,neighbors in {node:nx.all_neighbors(G,node) for node in nodes}.items() for target in neighbors)
#     siblings = set(tuple(sorted(edge)) for siblings in [tuple(G.successors(node)) for node in nodes] if len(siblings) > 1 for edge in itertools.combinations(siblings, 2))
#     print('Neighbors: {}'.format(len(neighbors)))
#     print('Siblings: {}'.format(len(siblings)))
#     return neighbors.union(siblings)

    ########################################
    ### PARALLEL (SLOW)
    ########################################
    # def parallelConnectedNodes(self,G):
    #     #END: 47.9949529171 secods.
    #     #TRANS MATCHING: 8 out of 21261.0
    #     #ONTO MATCHING: 8 out of 297.0
    #     neighbors = set()
    #     siblings = set()
    #
    #     data = G.nodes()
    #     nnodes = len(data)
    #     chunksize = nnodes if nnodes < 120000 else 500
    #     nchunks = int(round(len(data) / float(chunksize)))
    #     chunks = self.chunks(data,chunksize)
    #     pool = mp.Pool(int(nchunks*2)+1)
    #
    #     self.log('=== CONNECTED NODES ===')
    #     self.log('Nodes: {} | Chunk Size {} | Nchunks {}'.format(len(data),chunksize,nchunks))
    #     for i,nodes in enumerate(chunks):
    #         self.log('PROCESS: {} of {} | {} nodes.'.format(i,nchunks,len(nodes)))
    #         proc_neighbors = pool.apply_async(_getNeighbors, args=[G,nodes])
    #         proc_siblings = pool.apply_async(_getSiblings, args=[G,nodes])
    #
    #         neighbors |= set(proc_neighbors.get())
    #         siblings |= set(proc_siblings.get())
    #
    #     self.log('Processing...')
    #     pool.close()
    #     pool.join()
    #
    #     self.log('Neighbors: {}'.format(len(neighbors)))
    #     self.log('Siblings: {}'.format(len(siblings)))
    #     return neighbors.union(siblings)

 ########################################
    ### PARALLEL (WRONG)
    ########################################

    # def _parallelNeighbors(self,G,chunks):
    #     self.log('neighbors...')
    #     r1 = Parallel(n_jobs=5)(delayed(_getNeighbors)(G,nodes) for nodes in chunks)
    #     neighbors = set([n for l in r1 for n in l])
    #     return neighbors
    #
    # def _parallelSiblings(self,G,chunks):
    #     self.log('siblings...')
    #     r2 = Parallel(n_jobs=5)(delayed(_getSiblings)(G,nodes) for nodes in chunks)
    #     siblings = set([n for l in r2 for n in l])
    #     return siblings

    # def _parallelConnectedEdges(self, G, chunks):
    #     self.log('connected...')
    #     r = Parallel(n_jobs=20)(delayed(_getConnectedEdges)(G,nodes) for nodes in chunks)
    #     connected = set([n for l in r for n in l])
    #     return connected
    #
    # def parallelConnectedNodes(self,G):
    #     self.log('=== CONNECTED NODES ===')
    #     nodes = G.nodes()
    #     chunksize = len(nodes)/100
    #     chunks = self.chunks(nodes,chunksize)
    #     self.log('Nodes: {} | Chunks: {}'.format(len(nodes),len(nodes)/chunksize))
    #
    #     connected_edges = self._parallelConnectedEdges(G,chunks)
    #     self.log('DONE')
    #     return connected_edges
    #
    #     # #2.1248190403 secods
    #     # #TRANS MATCHING: 7 out of 21261.0
    #     # #ONTO MATCHING: 7 out of 125.0
    #     # self.log('=== CONNECTED NODES ===')
    #     # data = G.nodes()
    #     # chunksize = len(data)/3
    #     # chunks = self.chunks(data,chunksize)
    #     #
    #     # self.log('Nodes: {} | Chunks: {}'.format(len(data),len(data)/chunksize))
    #     # r1 = Parallel(n_jobs=2)(delayed(_getNeighbors)(G,nodes) for nodes in chunks)
    #     # neighbors = set([n for l in r1 for n in l])
    #     #
    #     # r2 = Parallel(n_jobs=2)(delayed(_getSiblings)(G,nodes) for nodes in chunks)
    #     # siblings = set([n for l in r2 for n in l])
    #     #
    #     # self.log('Neighbors: {}'.format(len(neighbors)))
    #     # self.log('Siblings: {}'.format(len(siblings)))
    #     # self.log('DONE')
    #     #
    #     # return neighbors.union(siblings)

    # def getEdgesByCommonChildren(self, G, nodes):
    #     pairs = []
    #     nodes = list(set(nodes))
    #     self.log('# nodes: {}'.format(len(nodes)))
    #     for node in nodes:
    #         ps = G.predecessors(node)
    #         ### If there are more than 1 parent,
    #         ### they they are connected by the same child 'node'
    #         if len(ps) > 1:
    #             pairs.extend([s for s in itertools.combinations(ps, 2)])
    #     return list(set(pairs))
    #
    # def getEdgesByShortestPath(self, G, threshold):
    #     try:
    #         return set(tuple(sorted(source,target)) for source,paths in nx.shortest_path(G).items() for target,path in paths.items() if source != target and (len(path)-1) >= 1 and (len(path)-1) <= threshold)
    #         #self.log('ALLPATHS-HOP{}: \n{}'.format(threshold,allpaths))
    #     except:
    #         return set()
    #
    # def getSiblingsOnly(self, G, edges):
    #     siblings = []
    #     for edge in edges:
    #         s = edge[0]
    #         t = edge[1]
    #         if self.nodesShareSameParents(G, s, t):
    #             siblings.append((s,t))
    #     return siblings
    #

    # def _getNeighbors(self, args):
    #     G, source = args
    #     return set(tuple(sorted((source,target))) for target in nx.all_neighbors(G,source))
    #
    # def _getSiblings(self, args):
    #     G, source = args
    #     return set(tuple(sorted(edge)) for edge in itertools.combinations(G.successors(source), 2))
    #
    # def parallel(self, G):
    #     num_cores = multiprocessing.cpu_count()
    #     pool = mp.ProcessingPool(num_cores)
    #
    #     neighbors = pool.map(self._getNeighbors, [(G,node) for node in G.nodes()])
    #     self.log('# neighbors: {}'.format(len(neighbors)))
    #
    #     siblings = pool.map(self._getSiblings, [(G,node) for node in G.nodes()])
    #     self.log('# siblings: {}'.format(len(siblings)))
    #
    #     return set(neighbors).union(set(siblings))

    # ### @deprecated
    # def getSiblings_old(self, G,nodes):
    #     pairs = []
    #     tmp = G.edges()
    #     for i,s in enumerate(nodes):
    #         for t in nodes[i+1:]:
    #             if self.nodesShareSameParents(G, s, t):
    #                 if (s,t) not in tmp and (t,s) not in tmp:
    #                     pairs.append((s,t))
    #     return pairs
    #
    # def numberOfUndirectedEdges(self, G):
    #     n = nx.number_of_nodes(G)
    #     return n * (n-1) / 2




