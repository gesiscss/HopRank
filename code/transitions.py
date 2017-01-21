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

#############################################################################
# Constants
#############################################################################
LOGFILE = '<path>/logTransitions-<name>.txt'
DUMMIES = ['INIT','END']
CONNECTED_SHORTEST_PATH = 1

class Transitions(Utils):

    #############################################################################
    # CONSTRUCTOR
    #############################################################################
    def __init__(self,fnonto,source,fnsource,results,type,filter,sizepath):
        self.fnonto = fnonto
        self.source = source
        self.fnsource = fnsource
        self.results = results
        self.output = os.path.join(results,'transitions')
        self.type = type
        self.filter = filter
        self.sizepath = sizepath
        self.name = self._getName()
        self.logfile = LOGFILE.replace('<path>',self.output).replace('<name>',self.name)

    #############################################################################
    # Functions Handlers
    #############################################################################

    def _getName(self):
        a=''
        b=''
        c=''

        ### Filter: all data or specific ontology
        if self.filter is None:
            a = '-alldata'
        else:
            for k,v in self.filter.items():
                a = '{}-{}:{}'.format(a,k,v)

        ### Type: Within or Across Ontologies
        if self.type == 'ontology':
            b ='-across_ontologies'
        elif self.type == 'concept':
            b = '-within_ontologies'

        ### Sizepath: number os HOPs between source and target node
        if self.sizepath == 0:
            c = '-directlink'
        else:
            c = '-{}indirectlinks'.format(self.sizepath)

        return '{}{}{}{}'.format(self.source,a,b,c)

    def _getFilename(self,postfix,ext):
        return os.path.join(self.output,'{}-{}.{}'.format(self.name,postfix,ext))

    def getConnectedNodesFileName(self):
        return self._getFilename('connected_nodes','p')

    #############################################################################
    # PATH POSITION VS DURATION
    #############################################################################

    def testTransitionsAndStructureLinks(self):
        G = nx.DiGraph()
        G.add_edge('A','B')
        G.add_edge('B','C')
        G.add_edge('E','C')
        G.add_edge('C','D')
        G.add_edge('C','F')
        G.add_edge('C','K')
        G.add_edge('G','I')
        G.add_edge('G','H')
        self.log('=== TEST ===\n{}\n=== END TEST ==='.format(self.getConnectedNodes(G)))

    def transitionsAndStructureLinks(self):
        onto = OntoParser(self.fnonto,self.results)
        tran = TransitionParser(self.source,self.fnsource,self.results,self.filter,self.type,self.sizepath)

        Gontology = onto.loadOntologyGraph()
        Gsessions = tran.getTransitionGraph()

        ### Removing edges from and to START and INIT (dummy states)
        for dummy in DUMMIES:
            Gsessions.remove_node(dummy)

        match = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0

        self.log('ONTOLOGY: \n{}'.format(Gontology.nodes()[:5]))
        self.log('TRANSITIONS: \n{}'.format(Gsessions.nodes()[:5]))

        ### Connected nodes
        fn = self.getConnectedNodesFileName()
        if self.exists(fn):
            connected_edges = self.loadPickle(fn)
            self.log('Connected Nodes Loaded! {}'.format(fn))
        else:
            connected_edges = self.parallelConnectedNodes(Gontology)
            #connected_edges = self.getConnectedNodes(Gontology)
            self.savePickle(connected_edges,fn)
            self.log('Connected Nodes Saved! {}'.format(fn))
        self.log('{} CONNECTED EDGES TO CHECK'.format(len(connected_edges)))

        ### Looking for matchings with transitions
        for edge in connected_edges:

            ### looking for matchings
            source = urllib.unquote(urllib.unquote(edge[0]))
            target = urllib.unquote(urllib.unquote(edge[1]))

            if source in Gsessions and target in Gsessions:
                flag = 1
                insession = self.areNodesNeighbors(Gsessions,source,target)
            else:
                source2 = source.split('/')[-1]
                target2 = target.split('/')[-1]

                if source2 in Gsessions and target2 in Gsessions:
                    flag = 2
                    insession = self.areNodesNeighbors(Gsessions,source2,target2)
                else:
                    flag = 3
                    insession = False

            ### Showing some examples
            if flag == 1 and counter1 < 5:
                self.log('(*) {} -> {} : {}'.format(source,target,insession))
                counter1 += 1
            elif flag == 2 and counter2 < 5:
                self.log('(**) {} -> {} : {}'.format(source2,target2,insession))
                counter2 += 1
            elif flag == 3 and counter3 < 5:
                self.log('(***) {}({}) -> {}({}) (DO NOT EXIST)'.format(source,source2,target,target2))
                counter3 += 1
            elif flag not in [1,2,3]:
                self.log('This is weird')

            ### counting up matchings
            match += int(insession)

        self.log(self.name)

        ### Summary Transitions
        total = float(Gsessions.number_of_edges())
        dismatch = total - match
        self.log('=== TRANSITIONS ===')
        self.log('TRANS MATCHING: {} out of {} ({}%)'.format(match,total,(match*100)/total))
        self.log('TRANS DIS-MATCHING: {} out of {} ({}%)'.format(dismatch,total,(dismatch*100)/total))

        ### Summary Transitions
        total = float(len(connected_edges))
        dismatch = total - match
        self.log('=== ONTOLOGY ===')
        self.log('ONTO MATCHING: {} out of {} ({}%)'.format(match,total,(match*100)/total))
        self.log('ONTO DIS-MATCHING: {} out of {} ({}%)'.format(dismatch,total,(dismatch*100)/total))

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
    filter = utils.getParameter(6)      # "{'ontology':'<ontology_name>'}" or "{'ontologies':['<ontology_name>','<ontology_name>']}" or None
    sizepath = utils.getParameter(7)    # integer (number of hops between source and target nodes)

    cp = Transitions(fnonto,source,fnsource,results,type,filter,sizepath)
    cp.loginit()
    cp.transitionsAndStructureLinks()
    cp.logend()
