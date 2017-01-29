__author__ = 'espin'

#############################################################################
# Dependences Local
#############################################################################
import utils
from utils import Utils
from ontologyParser import OntoParser
from transitionParser import TransitionParser

#############################################################################
# Dependences System
#############################################################################
import os
import time
import urllib
import networkx as nx
import sys
import pandas as pd

#############################################################################
# Constants
#############################################################################
LOGFILE = '<path>/log_transitions_<name>.txt'
CONNECTED_SHORTEST_PATH = 1

class Transitions(Utils):

    #############################################################################
    # CONSTRUCTOR
    #############################################################################
    def __init__(self,fnonto,source,fnsource,results,type,filterin,filterout,sizepath):
        self.fnonto = fnonto
        self.source = source
        self.fnsource = fnsource
        self.results = results
        self.filterin = filterin
        self.filterout = filterout
        self.year = self.getYearFromFileName(fnsource)
        self.output = self.generatePath(self.results,self.year,self.source,self.filterin,self.filterout)
        self.type = type
        self.sizepath = sizepath
        self.name = self._getName()
        self.logfile = LOGFILE.replace('<path>',self.output).replace('<name>',self.name)

    #############################################################################
    # Functions Handlers
    #############################################################################

    def _getName(self):
        a='FilterIn'
        c=''
        d=''

        ### Filterin: all data or specific ontology
        if self.filterin is None:
            a = 'alldata'.format(a)
        else:
            for k,v in self.filterin.items():
                a = '{}_{}_{}'.format(a,k,v)
                a = a.replace(' ','_')
            if len(self.filterin.keys()) == 1:
                a = a.replace('FilterIn_ontology_','')

        ### Type: Within or Across Ontologies
        if self.type == 'ontology':
            c ='across_ontos'
        elif self.type == 'concept':
            c = 'within_ontos'

        ### Sizepath: number os HOPs between source and target node
        if self.sizepath == 0:
            d = 'directpath'
        else:
            d = '{}HOP'.format(self.sizepath)

        return '{}_{}_{}_{}'.format(self.source[0],a,c,d)

    def _getFilename(self,prefix,ext):
        return os.path.join(self.output,'{}_{}.{}'.format(prefix,self.name,ext))

    def getTransitionsOverlapLinksFileName(self):
        return self._getFilename('transitions_overlap_links','gpickle')

    def getTransitionWeightsDistributionPlotFileName(self):
        return self._getFilename('transition_weight_distribution','png')

    #############################################################################
    # PATH POSITION VS DURATION
    #############################################################################

    def testTransitionsAndStructureLinks(self):
        # G = nx.DiGraph()
        # G.add_edge('A','B')
        # G.add_edge('B','C')
        # G.add_edge('E','C')
        # G.add_edge('C','D')
        # G.add_edge('C','F')
        # G.add_edge('C','K')
        # G.add_edge('G','I')
        # G.add_edge('G','H')
        # self.log('=== TEST ===\n{}\n=== END TEST ==='.format(self.getConnectedNodes(G)))

        # onto = OntoParser(self.fnonto,self.results)
        # Gontology = onto.loadOntologyGraph()
        #
        # node = 'http://purl.obolibrary.org/obo/DRON_00713136'
        # print '{} in graph: {}'.format(node,node in Gontology)
        #
        # node = 'DRON_00713136'
        # print '{} in graph: {}'.format(node,node in Gontology)
        #
        # node = '702231'
        # print '{} in graph: {}'.format(node,node in Gontology)
        #
        # node = '1000030'
        # print '{} in graph: {}'.format(node,node in Gontology)

        tran = TransitionParser(self.source,self.fnsource,self.results,self.filterin,self.filterout,self.type,self.sizepath)
        Gsessions = tran.getTransitionGraph()
        tran.removeDummies(Gsessions)
        tran.removeEdgesByWeight(Gsessions,1)

    def transitionWeightsDistribution(self):
        fn = self.getTransitionsOverlapLinksFileName()
        if self.exists(fn):
            Gsessions = self.loadGraph(fn)
        else:
            Gsessions = self.transitionsAndStructureLinks()

        weights = set([e[2]['weight'] for e in Gsessions.edges(data=True) if e[2]['valid']])
        weights = sorted(weights)

        data = {'weights':weights, 'match':[sum([1 for e in Gsessions.edges(data=True) if e[2]['valid'] and e[2]['match'] and e[2]['weight']==w]) for w in weights], 'mismatch':[sum([1 for e in Gsessions.edges(data=True) if e[2]['valid'] and not e[2]['match'] and e[2]['weight']==w]) for w in weights]}
        df = pd.DataFrame(data, columns = data.keys())

        tranmatch = sum([e[2]['weight'] for e in Gsessions.edges(data=True) if e[2]['valid'] and e[2]['match']])
        tranmismatch = sum([e[2]['weight'] for e in Gsessions.edges(data=True) if e[2]['valid'] and not e[2]['match']])
        legend = ['Match ({})'.format(tranmatch),'Mismatch ({})'.format(tranmismatch)]

        ### PLOT
        fn = self.getTransitionWeightsDistributionPlotFileName()
        self.plotGroupBars(df=df,
                           title='{}'.format('ALL ONTOLOGIES' if self.filterin is None else '{} CONCEPTS'.format(self.filterin['ontology'])),
                           xlabel='# Transitions (Edge Weights)',
                           fn=fn,
                           labelkey='weights',
                           groups=['match','mismatch'],
                           ylog=True,
                           xticks=True,
                           posdelta=True,
                           legend=legend)
        return

    def transitionsAndStructureLinks(self):

        onto = OntoParser(self.fnonto,self.results)
        Gontology = onto.loadOntologyGraph()

        fn = self.getTransitionsOverlapLinksFileName()
        if self.exists(fn):
            Gsessions = self.loadGraph(fn)
        else:
            tran = TransitionParser(self.source,self.fnsource,self.results,self.filterin,self.filterout,self.type,self.sizepath)
            Gsessions = tran.getTransitionGraph()
            Gsessions = tran.removeDummies(Gsessions)
            Gsessions = tran.removeEdgesByWeight(Gsessions,1)
            self.log('ONTOLOGY: \n{}'.format(Gontology.nodes()[:5]))
            self.log('{} EDGES IN GRAPH ONTOLOGY'.format(Gontology.number_of_edges()))
            self.log('TRANSITIONS (no dummies, edge weights > 1) (sessions): \n{}'.format(Gsessions.nodes()[:5]))
            self.log('{} EDGES IN GRAPH SESSIONS'.format(Gsessions.number_of_edges()))

            counter1 = 0
            counter2 = 0
            noedges = 0

            nx.set_node_attributes(Gsessions, 'valid', False)

            ### Looking for matchings with transitions
            for edge in Gsessions.edges(data=True):

                ### looking for matchings
                source = edge[0]
                target = edge[1]
                insession = False
                Gsessions.node[source]['valid'] = source in Gontology
                Gsessions.node[target]['valid'] = target in Gontology

                if source in Gontology and target in Gontology:
                    insession = self.areNodesConnected(Gontology,source,target)
                    Gsessions[source][target]['valid'] = True
                    Gsessions[source][target]['match'] = insession

                else:

                    self.log('Some states do not exist in the ontology.')
                    self.log('source: {} ({})'.format(source,source in Gontology))
                    self.log('target: {} ({})'.format(target,target in Gontology))
                    Gsessions[source][target]['valid'] = False
                    Gsessions[source][target]['match'] = False
                    noedges += 1
                    continue

                ### Showing some examples
                if insession and counter1 < 5:
                    self.log('(*) {} -> {} : {}'.format(source,target,insession))
                    counter1 += 1
                elif not insession and counter2 < 5:
                    self.log('(**) {} -> {} : {}'.format(source,target,insession))
                    counter2 += 1

            ### Save new graph (edges are now labeled: remove, match. Also dummy and weight=1 edges are removed)
            self.log('Transitions: \n{}'.format(nx.info(Gsessions)))
            counter = 0
            for n in Gsessions.nodes(data=True):
                if not n[1]['valid']:
                    Gsessions.remove_node(n[0])
                    counter += 1
            self.log('Transitions after removing {} not existing nodes in ontology: \n{}'.format(counter,nx.info(Gsessions)))
            self.saveGraph(Gsessions,fn)

        self.log(self.name)

        ### Summary Transitions
        total = float(Gsessions.number_of_edges())
        transitions = sum([edge[2]['weight'] for edge in Gsessions.edges(data=True) if edge[2]['valid']])
        match = sum([1 for edge in Gsessions.edges(data=True) if edge[2]['match'] and edge[2]['valid']])
        mismatch = sum([1 for edge in Gsessions.edges(data=True) if not edge[2]['match'] and edge[2]['valid']])

        self.log('=== SESSIONS ===')
        self.log('{} edges removed (some nodes do not exist in the ontology'.format(noedges))
        self.log('Total edges: {} | Total transitions: {}'.format(total,transitions))

        self.log('=== TRANSITIONS OVERLAP ===')
        self.log('TRANS MATCHING: {} out of {} ({}%)'.format(match,total,(match*100)/float(total)))
        self.log('TRANS DIS-MATCHING: {} out of {} ({}%)'.format(mismatch,total,(mismatch*100)/float(total)))

        ### Summary Ontology
        total = Gontology.number_of_edges()
        mismatch = total - match
        self.log('=== ONTOLOGY OVERLAP (approximate total, it should be bigger considering siblings) ===')
        self.log('ONTO MATCHING: {} out of {} ({}%)'.format(match,total,(match*100)/float(total)))
        self.log('ONTO DIS-MATCHING: {} out of {} ({}%)'.format(mismatch,total,(mismatch*100)/float(total)))

        self.log('Valid Transitions:')
        self.log(nx.info(Gsessions))

        self.log('Ontology:')
        self.log(nx.info(Gontology))

        return Gsessions



#############################################################################
# Main
# e.g.:
#############################################################################
if __name__ == '__main__':
    fnonto = utils.getParameter(1)      # ontology file
    source = utils.getParameter(2)      # clickstream, apirequests
    fnsource = utils.getParameter(3)    # 'transitions log file
    results = utils.getParameter(4)     # path pointing to results folder
    type = utils.getParameter(5)        # ontology (across ontologies), concept (within ontologies)
    filterin = utils.getParameter(6)   # "{'ontology':'<ontology_name>'}" or "{'ontologies':['<ontology_name>','<ontology_name>']}" or None
    filterout = utils.getParameter(7)  # "{'request_action':'Browse Ontology Class Tree'}" or None
    sizepath = utils.getParameter(8)    # integer (number of hops between source and target nodes)

    cp = Transitions(fnonto,source,fnsource,results,type,filterin,filterout,sizepath)
    cp.loginit()
    #cp.testTransitionsAndStructureLinks()
    cp.transitionsAndStructureLinks()
    cp.transitionWeightsDistribution()
    cp.logend()




    # def transitionsAndStructureLinksOLD(self):
    #     onto = OntoParser(self.fnonto,self.results)
    #     tran = TransitionParser(self.source,self.fnsource,self.results,self.filter,self.type,self.sizepath)
    #
    #     Gontology = onto.loadOntologyGraph()
    #     Gsessions = tran.getTransitionGraph()
    #
    #     ### Removing edges from and to START and INIT (dummy states)
    #     for dummy in DUMMIES:
    #         Gsessions.remove_node(dummy)
    #     ### Remove edges with weight < 10
    #
    #
    #     match = 0
    #     counter1 = 0
    #     counter2 = 0
    #     counter3 = 0
    #
    #     self.log('ONTOLOGY: \n{}'.format(Gontology.nodes()[:5]))
    #     self.log('TRANSITIONS: \n{}'.format(Gsessions.nodes()[:5]))
    #     self.log('{} EDGES IN GRAPH'.format(Gontology.number_of_edges()))
    #
    #     ### Connected nodes
    #     fn = self.getConnectedNodesFileName()
    #     if self.exists(fn):
    #         connected_edges = self.loadPickle(fn)
    #         self.log('Connected Nodes Loaded! {}'.format(fn))
    #     else:
    #         connected_edges = self.parallelConnectedNodes(Gontology)
    #         #connected_edges = self.getConnectedNodes(Gontology)
    #         self.savePickle(connected_edges,fn)
    #         self.log('Connected Nodes Saved! {}'.format(fn))
    #
    #
    #     self.log('{} CONNECTED EDGES TO CHECK'.format(len(connected_edges)))
    #
    #     ### Looking for matchings with transitions
    #     for edge in connected_edges:
    #
    #         ### looking for matchings
    #         source = urllib.unquote(urllib.unquote(edge[0]))
    #         target = urllib.unquote(urllib.unquote(edge[1]))
    #
    #         if source in Gsessions and target in Gsessions:
    #             flag = 1
    #             insession = self.areNodesNeighbors(Gsessions,source,target)
    #         else:
    #             source2 = source.split('/')[-1]
    #             target2 = target.split('/')[-1]
    #
    #             if source2 in Gsessions and target2 in Gsessions:
    #                 flag = 2
    #                 insession = self.areNodesNeighbors(Gsessions,source2,target2)
    #             else:
    #                 flag = 3
    #                 insession = False
    #
    #         ### Showing some examples
    #         if flag == 1 and counter1 < 5:
    #             self.log('(*) {} -> {} : {}'.format(source,target,insession))
    #             counter1 += 1
    #         elif flag == 2 and counter2 < 5:
    #             self.log('(**) {} -> {} : {}'.format(source2,target2,insession))
    #             counter2 += 1
    #         elif flag == 3 and counter3 < 5:
    #             self.log('(***) {}({}) -> {}({}) (DO NOT EXIST)'.format(source,source2,target,target2))
    #             counter3 += 1
    #         elif flag not in [1,2,3]:
    #             self.log('This is weird')
    #
    #         ### counting up matchings
    #         match += int(insession)
    #
    #     self.log(self.name)
    #
    #     ### Summary Transitions
    #     total = float(Gsessions.number_of_edges())
    #     dismatch = total - match
    #     self.log('=== TRANSITIONS ===')
    #     self.log('TRANS MATCHING: {} out of {} ({}%)'.format(match,total,(match*100)/total))
    #     self.log('TRANS DIS-MATCHING: {} out of {} ({}%)'.format(dismatch,total,(dismatch*100)/total))
    #
    #     ### Summary Transitions
    #     total = float(len(connected_edges))
    #     dismatch = total - match
    #     self.log('=== ONTOLOGY ===')
    #     self.log('ONTO MATCHING: {} out of {} ({}%)'.format(match,total,(match*100)/total))
    #     self.log('ONTO DIS-MATCHING: {} out of {} ({}%)'.format(dismatch,total,(dismatch*100)/total))
