__author__ = 'espin'

#############################################################################
# Dependences
#############################################################################
import utils
import networkx as nx
import pandas
import os
from utils import Utils

#############################################################################
# Constants
#############################################################################
FIELD_SEP = ','
PARENT_SEP = '|'
COMPRESSION = 'gzip'
LOGFILE = '<path>/logOntologyParser-<name>.txt'

#############################################################################
# Class
#############################################################################

class OntoParser(Utils):

    def __init__(self, fn, results):
        self.fn = fn
        self.output = os.path.join(results,'ontologies')
        self.ontoname = self._getOntoName()
        self.logfile = LOGFILE.replace('<path>',self.output).replace('<name>',self.ontoname)

    #############################################################################
    # Functions Handlers
    #############################################################################
    def _getOntoName(self):
        try:
            tmp = self.fn.split('/')[-1].replace('.csv','').replace('.gz','')
        except:
            tmp = None
        return tmp

    def _getFilename(self,ext):
        return os.path.join(self.output,'{}.{}'.format(self.ontoname,ext))

    def getGraphFilename(self):
        return self._getFilename('gpickle')

    def getAdjacencyFilename(self):
        return self._getFilename('mtx')

    def getAdjacencyPlotFilename(self):
        return self._getFilename('pdf')


    #############################################################################
    # Functions
    #############################################################################

    ###
    # Generates a networkx Directed Graph
    # from ontology data classes metadata
    ###
    def createOntology(self):

        G = nx.DiGraph(name=self.ontoname)
        fn = self.getGraphFilename()

        if not self.exists(fn):

            df = pandas.read_csv(self.fn,sep=FIELD_SEP,compression=COMPRESSION)
            df = df[['Class ID', 'Parents']]

            for index, row in df.iterrows():
                parents = str(row['Parents']).split(PARENT_SEP)

                ### add node (class)
                if row['Class ID'] not in G:
                    G.add_node(row['Class ID'],classid=row['Class ID'])

                for parent in parents:

                    if parent != 'nan':

                        ### add parent node
                        if parent not in G:
                            G.add_node(parent, classid=parent)

                        ### add link
                        G.add_edge(parent,row['Class ID'])

            self.saveGraph(G,self.getGraphFilename())
            self.saveAdjacencyMatrix(G,self.getAdjacencyFilename())
        else:
            self.log('Loading Ontology Graph: {}'.format(fn))
            G = self.loadGraph(fn)

        self.log('number of nodes: {}'.format(G.number_of_nodes()))
        return G

    def loadOntologyGraph(self):
        fn = self.getGraphFilename()
        if not self.exists(fn):
            return self.createOntology()
        self.log('Loading Ontology Graph: {}'.format(fn))
        return self.loadGraph(fn)


#############################################################################
# Main
#############################################################################
if __name__ == '__main__':
    fn = utils.getParameter(1)      # GZIP file with ontology information
    results = utils.getParameter(2)  # path pointing to results

    op = OntoParser(fn,results)
    op.createOntology()
    op.log('END...')
