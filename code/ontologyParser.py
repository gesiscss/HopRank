__author__ = 'espin'

#############################################################################
# Dependences
#############################################################################
import utils
import networkx as nx
import pandas
import os

#############################################################################
# Constants
#############################################################################
FIELD_SEP = ','
PARENT_SEP = '|'
COMPRESSION = 'gzip'

#############################################################################
# Functions Handlers
#############################################################################
def _getOntoName(fn):
    try:
        tmp = fn.split('/')[-1].replace('.csv','')
    except:
        tmp = None
    return tmp

def _getFilename(ontoname,ext,output):
    return os.path.join(output,'{}.{}'.format(ontoname,ext))

def getGraphFilename(ontoname,output):
    return _getFilename(ontoname,'gpickle',output)

def getAdjacencyFilename(ontoname,output):
    return _getFilename(ontoname,'mtx',output)

def getAdjacencyPlotFilename(ontoname,output):
    return _getFilename(ontoname,'pdf',output)


#############################################################################
# Functions
#############################################################################

###
# Generates a networkx Directed Graph
# from ontology data classes metadata
###
def createOntology(fn,output):

    ontoname = _getOntoName(fn)
    G = nx.DiGraph(name=ontoname)

    if not utils.exists(getGraphFilename(ontoname,output)):

        df = pandas.read_csv(fn,sep=FIELD_SEP,compression=COMPRESSION)
        df = df[['Class ID', 'Parents']]

        for index, row in df.iterrows():
            #print(index,row['Parents'])
            parents = str(row['Parents']).split(PARENT_SEP)

            ### add node (class)
            if row['Class ID'] not in G:
                G.add_node(row['Class ID'],classid=row['Class ID'])

            for parent in parents:

                print(parent)

                if parent != 'nan':

                    ### add parent node
                    if parent not in G:
                        G.add_node(parent, classid=parent)

                    ### add link
                    G.add_edge(parent,row['Class ID'])

        utils.saveGraph(G,getGraphFilename(G.graph['name'],output))
    else:
        G = utils.loadGraph(getGraphFilename(ontoname,output))

    print('number of nodes: {}'.format(G.number_of_nodes()))
    utils.saveAdjacencyMatrix(G,getAdjacencyFilename(G.graph['name'],output))



#############################################################################
# Main
#############################################################################
if __name__ == '__main__':
    fn = utils.getParameter(1)
    output = utils.getParameter(2)
    createOntology(fn,output)
