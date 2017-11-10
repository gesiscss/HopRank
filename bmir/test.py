import sys
import pandas as pd
import math, string, sys, fileinput
import snap
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing

class Entropy():
    #!/usr/bin/python
    #
    # Stolen from Ero Carrera
    # http://blog.dkbza.org/2007/05/scanning-data-for-entropy-anomalies.html

    def range_bytes (self): return range(256)
    def range_printable(self): return (ord(c) for c in string.printable)
    def H(self,data, iterator=range_bytes):
        if not data:
            return 0
        entropy = 0
        for x in iterator():
            p_x = float(data.count(chr(x)))/len(data)
            if p_x > 0:
                entropy += - p_x*math.log(p_x, 2)
        return entropy

def test_entropy():
    en = Entropy()

    for str in ['gargleblaster', 'tripleee', 'magnus', 'lkjasdlk','aaaaaaaa', 'sadfasdfasdf', '7&wS/p(','bb','bbbbbbbbbb','aaaaaaaaab','abababababab','abcabcabcabcabcabc']:
        print ("%s: %f" % (str, en.H(str, en.range_printable)))

def test_break():

    for i in range(1,10):
        print i
        for j in range(100,105):
            print j
            if j == 102:
                break

def test_df_loc():
    d = {'one' : pd.Series([1., 2., 3., 1.]),
         'two' : pd.Series([10., 20., 30., 40.]),
         'three' : pd.Series([100., 200., 300., 400.])}

    df = pd.DataFrame(d)
    df = df.sort(['one'],ascending=False)

    print df
    print df.index
    print df.columns
    print df.size
    print df.shape
    print df.loc[0]
    print df.iloc[0]

    dfg = df.groupby(['one'], as_index=False)

    print dfg
    print dfg.head()
    print ''
    for name,group in dfg:
        print name
        print group
        print ''


    print ''
    for name,group in dfg:
        for index, row in group.iterrows():
            print index
            print row
            print ''

    print 'indices:'
    print dfg.apply(lambda x: x.index)

    print 'row 0:'
    print [row for name,group in dfg for index, row in group.iterrows() if index == 3][0]


    #    one  three   two
    # 2  3.0  300.0  30.0
    # 1  2.0  200.0  20.0
    # 0  1.0  100.0  10.0
    # 3  1.0  400.0  40.0

    #    one  three   two
    # 2  3.0  300.0  30.0
    # 1  2.0  200.0  20.0
    # 0  1.0  100.0  10.0
    # 3  1.0  400.0  40.0

def test_hops():
    G = nx.Graph()
    G.add_edge('A','B')
    G.add_edge('A','C')
    G.add_edge('C','D')
    G.add_edge('C','E')

    ### ONTOLOGY (directed) as SNAP GRAPH
    H = snap.TNGraph.New()
    print('snap: Directed graph created!')

    for node in G.nodes():
        H.AddNode(G.nodes().index(node))
        print('{}: {}'.format(G.nodes().index(node), node))
    print('snap: {} nodes added!'.format(H.GetNodes()))

    for s,t in G.edges():
        H.AddEdge(G.nodes().index(s),G.nodes().index(t))
    print('snap: {} edges added!'.format(H.GetEdges()))


    #classid

    edges = {}
    ### LOOKING FOR k-HOPS
    for k in range(1,5+1):
        print('=== HOP {} ==='.format(k))
        edges[k] = []

        for source in [n.GetId()  for n in H.Nodes()]:
            print('source node: {} ({})'.format(G.nodes()[source], source))

            NodeVec = snap.TIntV()
            nodes = snap.GetNodesAtHop(H, source, k+1, NodeVec, False)

            if nodes > 0:
                print('{} nodes at {}-HOP distance: '.format(nodes,k))
            edges[k].extend( [(source,target) for target in NodeVec] )

        if len(edges[k]) > 0:
            print('{} edges in {}-HOP'.format(len(edges[k]),k))

def test_root_node():
    ROOT = 'root'
    G = nx.DiGraph()
    G.add_edge(ROOT, 'b')
    G.add_edge(ROOT, 'e')
    G.add_edge('b', 'c')
    G.add_edge('b', 'd')
    G.add_edge('d', 'g')
    G.add_edge('e', 'f')
    G.add_edge('h', 'f')
    G.add_edge('i', 'j')

    # if ROOT not in G:
    roots = [n for n in G.nodes() if len(G.predecessors(n)) == 0 and n != ROOT]
    for n in roots:
        G.add_edge(ROOT, n)
        print('new edge: {} -> {}'.format(ROOT,n))

    print('summary:')
    print(G.edges())

def to_undirect(graph,weighted=True):
    H = nx.Graph()
    H.add_nodes_from(graph.nodes())
    if weighted:
        H.add_edges_from(graph.edges_iter(), weight=0)
        for u, v, d in graph.edges_iter(data=True):
            H[u][v]['weight'] += d['weight']
    else:
        H.add_edges_from(graph.edges_iter())
    return H

def _overlap_khops(edge, ontology_graph, k):
    distance = nx.shortest_path_length(ontology_graph, source=edge[0], target=edge[1])
    overlap = distance == k
    return {'edge':edge, 'overlap':overlap}

def test_hop_overlap(k=1):
    print('init')
    # 1. get networks
    ontology_graph = nx.DiGraph(name='ontology')
    ontology_graph.add_edge('A', 'B')
    ontology_graph.add_edge('B', 'C')
    ontology_graph.add_edge('B', 'D')
    ontology_graph.add_edge('B', 'E')
    ontology_graph.add_edge('D', 'F')
    ontology_graph.add_edge('D', 'G')

    transition_graph = nx.DiGraph(name='transitions')
    transition_graph.add_edge('A', 'B', weight=4)
    transition_graph.add_edge('B', 'C', weight=4)
    transition_graph.add_edge('C', 'D', weight=1)
    transition_graph.add_edge('B', 'D', weight=1)
    transition_graph.add_edge('D', 'G', weight=1)
    transition_graph.add_edge('B', 'A', weight=10)
    transition_graph.add_edge('G', 'A', weight=5)

    # 2. make sure undirected
    ontology_graph = to_undirect(ontology_graph,False)
    print(nx.info(ontology_graph))
    transition_graph = to_undirect(transition_graph,True)
    print(nx.info(transition_graph))
    print(transition_graph.size(weight='weight'))

    # 3. looking for ovelaps in k hop
    num_cores = multiprocessing.cpu_count()
    print('num cores: {}'.format(num_cores))
    results = Parallel(n_jobs=num_cores)(delayed(_overlap_khops)(edge, ontology_graph, k) for edge in transition_graph.edges(data=True))

    print('results: {}'.format(len(results)))
    print(results[0])

    onto_nodes = ontology_graph.number_of_nodes()
    trans_nodes = transition_graph.number_of_nodes()

    edges_visited = [data['edge'] for data in results if data['overlap']]
    nodes_visited = [(edge[0], edge[1]) for edge in edges_visited]
    nodes_visited = zip(*nodes_visited)
    nodes_visited = set([a for b in nodes_visited for a in b])
    nodes_visited = len(nodes_visited)

    edges_overlap = sum([1 for data in results if data['overlap']])
    onto_edges = ontology_graph.number_of_edges()
    trans_edges = transition_graph.number_of_edges()

    multiedges_overlap = sum([data['edge'][2]['weight'] for data in results if data['overlap']])
    multiedges = sum([data['edge'][2]['weight'] for data in results])

    # 4. scores
    scores = {}
    scores['structure_nodes'] = {'raw': onto_nodes, 'overlap_percentage': 0 if onto_nodes == 0 else nodes_visited / float(onto_nodes)}
    scores['transition_nodes'] = {'raw': trans_nodes, 'overlap_percentage': 0 if trans_nodes == 0 else nodes_visited / float(trans_nodes)}

    scores['structure_edges'] = {'raw': onto_edges, 'overlap_percentage': 0 if onto_edges == 0 else edges_overlap / float(onto_edges)}
    scores['transition_edges'] = {'raw': trans_edges, 'overlap_percentage': 0 if trans_edges == 0 else edges_overlap / float(trans_edges)}

    scores['transition_multiedges'] = {'raw': multiedges, 'overlap_percentage': 0 if multiedges == 0 else multiedges_overlap / float(multiedges)}

    scores['intersection'] = {'nnodes': nodes_visited, 'nedges': edges_overlap, 'nmultiedges': multiedges_overlap}
    scores['score_overlap_edges'] = 0
    scores['score_overlap_multiedges'] = 0

    print(scores)

if __name__ == '__main__':
    # test_entropy()
    # test_hops()
    # test_root_node()
    test_hop_overlap(3)