__author__ = 'espin'

#############################################################################
# Dependences
#############################################################################
import utils
import networkx as nx
import pandas as pd
import os
import sys

#############################################################################
# Constants
#############################################################################
FIELD_SEP = ','
PARENT_SEP = '|'
COMPRESSION = 'gzip'
DEFAULTTYPE = 'concept' #within ontology by default

#############################################################################
# Functions Handlers
#############################################################################

def getName(filter,type):
    a=''
    b=''
    if filter is None:
        a='-alldata'
    else:
        for k,v in filter.items():
            a = '{}-k:v'.format(a,k,v)

    if type == 'ontology':
        b='-across_ontologies'
    elif type == 'concept':
        b = '-within_ontologies'

    return 'clickstream-alldata{}{}'.format(a,b)

def _getFilename(name,ext,output):
    return os.path.join(output,'{}.{}'.format(name,ext))

def getGraphFilename(name,output):
    return _getFilename(name,'gpickle',output)

def getAdjacencyFilename(name,output):
    return _getFilename(name,'mtx',output)



#############################################################################
# Functions
#############################################################################
###
# Creates a graph from the clickstream data of users.
# The fulter param lets you filter out not needed info.
# If ontology-specific graph then: filter={'ontology':'<ontology_name>'}
# If ip-specific graph then: filter={'ip':'<ipaddress>'}
# If all data: filter=None
# Type: ontology (across ontologies), concept (within ontologies)
###
def createClickstream(fn,output,filter=None,type=DEFAULTTYPE):

    if 'ontology' in filter and type == 'ontology':
        print('WARNING: You have selected clickstreams across ontologies.')
        sys.exit(0)

    name = getName(filter,type)
    G = nx.DiGraph(name=name)

    if not utils.exists(getGraphFilename(name,output)):

        df = pd.read_csv(fn,sep=FIELD_SEP,compression=COMPRESSION)

        if filter is not None:
            for column_name, value in filter.items():
                df.loc[df[column_name] == value]

        previous_node = None
        for index, row in df.iterrows():
            current_node = row[type]

            ### add node (class)
            if current_node not in G:
                G.add_node(current_node,classid=current_node)

            ### add link
            if previous_node is not None:
                G.add_edge(previous_node,current_node,
                           timestamp=row['timestamp'],
                           req_id=row['req_id'],
                           ip=row['ip'],
                           action=row['action'],
                           request=row['request'],
                           request_action=row['request_action'],
                           statuscode=row['status_code'],
                           referer=row['referer'],
                           useragent=row['useragent'],
                           concept=row['concept'])
            else:
                previous_node = current_node

        utils.saveGraph(G,getGraphFilename(G.graph['name'],output))
    else:
        G = utils.loadGraph(getGraphFilename(name,output))

    print('number of nodes: {}'.format(G.number_of_nodes()))
    utils.saveAdjacencyMatrix(G,getAdjacencyFilename(name,output))



#############################################################################
# Main
#############################################################################
if __name__ == '__main__':
    fn = utils.getParameter(1)
    output = utils.getParameter(2)
    type = utils.getParameter(3) # ontology (across ontologies), concept (within ontologies)
    filter = utils.getParameter(4) # "{'ontology':'<ontology_name>'}"
    createClickstream(fn, output, filter, type)