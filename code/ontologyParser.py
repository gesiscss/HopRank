__author__ = 'espin'

#############################################################################
# Dependences
#############################################################################
import utils
import networkx as nx
import pandas
import os
from utils import Utils
import urllib

#############################################################################
# Constants
#############################################################################
FIELD_SEP = ','
PARENT_SEP = '|'
COMPRESSION = 'gzip'
LOGFILE = '<path>/log_parser_<name>.txt'
SOURCE = 'ontologies'

#############################################################################
# Class
#############################################################################

class OntoParser(Utils):

    def __init__(self, fn, results):
        self.fn = fn
        self.ontoname = self._getOntoName()
        self.year = self.getYearFromFileName(fn)
        self.results = results
        self.output = self.generateSimplePath(self.results,self.year,SOURCE)
        self.logfile = LOGFILE.replace('<path>',self.output).replace('<name>',self.ontoname)
        self.log('ONTLOGY NAME: {}'.format(self.ontoname))
        self.log('OUTPUT: {}'.format(self.output))

    #############################################################################
    # Functions Handlers
    #############################################################################
    def _getOntoName(self):
        try:
            tmp = self.fn.split('/')[-1].replace('.csv','').replace('.gz','').split('_')[0]
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
                node = self.getConceptId(row['Class ID'])
                parents = str(row['Parents']).split(PARENT_SEP)

                ### add node (class)
                if row['Class ID'] not in G:
                    G.add_node(node,classid=node)

                for parent in parents:

                    if parent.lower() != 'nan':

                        parent = self.getConceptId(parent)

                        ### add parent node
                        if parent not in G:
                            G.add_node(parent, classid=parent)

                        ### add link
                        G.add_edge(parent,node)

            self.saveGraph(G,self.getGraphFilename())
            self.saveAdjacencyMatrix(G,self.getAdjacencyFilename())
        else:
            self.log('Loading Ontology Graph: {}'.format(fn))
            G = self.loadGraph(fn)

        self.log('SUMMARY: \n{}'.format(nx.info(G)))
        return G

    def loadOntologyGraph(self):
        fn = self.getGraphFilename()
        if not self.exists(fn):
            return self.createOntology()
        self.log('Loading Ontology Graph: {}'.format(fn))
        return self.loadGraph(fn)

    def getConceptId(self, classid):
        name = None
        tmp = classid
        if tmp != 'http' and len(str(tmp)) > 0 and str(tmp).lower() != 'nan' and tmp != '':
            if 'http' in tmp:

                if 'Thesaurus' in tmp:
                    name = tmp
                else:
                    tmp = urllib.unquote(urllib.unquote(tmp))
                    key = '/ontology/'
                    if key in tmp:
                        s = tmp[tmp.index(key)+len(key):].split('/')
                        if len(s) == 1:
                            name = s[0] #ontology
                            self.log('len==1: {} : {} <<<<<<----------------------------- this must be an error?'.format(name,classid))
                        else:
                            name = s[1] #concept id
                    else:
                        name = tmp

            elif tmp.lower() != 'nan':
                name = tmp
        return name

#############################################################################
# Main
#############################################################################
if __name__ == '__main__':
    fn = utils.getParameter(1)       # GZIP file with ontology information
    results = utils.getParameter(2)  # path pointing to results

    op = OntoParser(fn,results)
    op.loginit()
    op.createOntology()
    op.logend()
