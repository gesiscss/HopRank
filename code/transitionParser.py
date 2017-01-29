__author__ = 'espin'

#############################################################################
# Dependences Local
#############################################################################
import utils
from utils import Utils

#############################################################################
# Dependences System
#############################################################################
from scipy.sparse import csr_matrix
import networkx as nx
import pandas as pd
import urllib
import sys
import os
import itertools

#############################################################################
# Constants
#############################################################################
FIELD_SEP = ','
PARENT_SEP = '|'
COMPRESSION = 'bz2'
DEFAULTTYPE = 'concept' #within ontology by default
LOGFILE = '<path>/log_parser_<name>.txt'
INNACTIVITY = 1800 #30min
TSFORMAT = '%Y-%m-%d %H:%M:%S'
DUMMYSTART = 'INIT'
DUMMYEND = 'END'
DUMMIES = [DUMMYSTART,DUMMYEND]
KEY = {'clickstream':'ip', 'apirequests':'apikey'}
MAX_LENGTH = 6


class TransitionParser(Utils):

    #############################################################################
    # CONSTRUCTOR
    #############################################################################
    def __init__(self,source,fn,results,filterin,filterout,type,sizepath):
        self.source = source
        self.results = results
        self.fn = fn
        self.filterin = filterin
        self.filterout = filterout
        self.year = self.getYearFromFileName(fn)
        self.output = self.generatePath(self.results,self.year,self.source,self.filterin,self.filterout)
        self.type = type  # concept, ontology (for generation of states)
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

    def getGraphFilename(self):
        return self._getFilename('graph','gpickle')

    def getAdjacencyFilename(self):
        return self._getFilename('adjacency','mtx')

    def getSessionsFilename(self):
        return self._getFilename('sessions','p')

    def getEntryPointsDataFileName(self):
        return self._getFilename('entry_points','p')

    def getExitPointsDataFileName(self):
        return self._getFilename('exit_points','p')

    def getEntryAndExitPointsPlotFileName(self):
        return self._getFilename('entry_exit_points','png')

    def getDistributionEdgeWeightsPlotFileName(self):
        return self._getFilename('distribution_edgeweights','png')

    def getCountsPositionVsDurationPlotFileName(self,session_length):
        return self._getFilename('position_vs_duration_{}nodesession'.format(session_length),'png')

    #############################################################################
    # TRANSITION GRAPH AND MATRIX
    #############################################################################
    ###
    # Creates a graph from the clickstream data of users.
    # * The filter param lets you filter out not needed info.
    #   If ontology-specific graph then: filter={'ontology':'<ontology_name>'}
    #   If ip-specific graph then: filter={'ip':'<ipaddress>'}
    #   If all data: filter=None
    # * Type: ontology (across ontologies), concept (within ontologies, then filter should specify an ontolgoy name)
    ###
    def createTransitionGraph(self):

        if self.filterin is not None and 'ontology' in self.filterin and self.type == 'ontology':
            self.log('WARNING: You have selected {} across ontologies. Exit.'.format(self.source))
            sys.exit(0)

        G = nx.DiGraph(name=self.name)
        G.add_node(DUMMYSTART)
        G.add_node(DUMMYEND)

        if not self.exists(self.getGraphFilename()):

            df = self.readServerLogData()

            ### generates sessions by users
            sessions = self.createSessions(df)

            ### creates a graph from sessions
            for key, obj in sessions.items():
                for sequence,actions in obj.items():
                    previous_node = None

                    for index,state in enumerate(actions):
                        current_node = self.getNodeName(state)

                        if current_node is not None:

                            ### Defining previous and current nodes (including dummy start and end.
                            if index == 0 or previous_node is None:
                                previous_node = DUMMYSTART
                            elif index == (len(actions)-1):
                                previous_node = current_node
                                current_node = DUMMYEND

                            if current_node != previous_node:

                                if current_node is None or previous_node is None:
                                    self.log('ERROR: NONE VALUE!!!')
                                    print(current_node)
                                    print(previous_node)
                                    sys.exit(0)

                                ### Creating nodes
                                if current_node not in G:
                                    G.add_node(current_node)

                                ### Creating an edge if it doesn't exist
                                if not G.has_edge(previous_node,current_node):
                                    G.add_edge(previous_node,current_node,weight=0.0)

                                ### Once the edge exists, increments weight
                                G[previous_node][current_node]['weight'] += 1.0
                                previous_node = current_node

            self.saveGraph(G,self.getGraphFilename())
            self.saveAdjacencyMatrix(G,self.getAdjacencyFilename())
        else:
            sessions = self.getSessions()
            self.log('Loading Transitions Graph: {}'.format(fn))
            G = self.loadGraph(self.getGraphFilename())

        self.log('SUMMARY (with DUMMIES)\n{}'.format(nx.info(G)))
        G.remove_nodes_from(DUMMIES)
        self.log('SUMMARY (without DUMMIES)\n{}'.format(nx.info(G)))

        ### The calculation of mode, min, max, etc. can be done faster using numpy ( i think )
        weights = [edge[2]['weight'] for edge in G.edges(data=True) if 'weight' in edge[2]]
        onepercent = len(sessions.keys()) * 1 / 100.
        self.log('\n=== SUMMARY WEIGHTS ===\n({} avg | {} median | {} mode | {} max | {} min | {} == 1 | {} > 1 | {} > 5 | {} > avg | {} > {} 1% {} | {} total {} )'.format(
            self.avg(weights), self.median(weights), self.mode(weights), max(weights), min(weights),
            len([w for w in weights if w == 1]),
            len([w for w in weights if w > 1]),
            len([w for w in weights if w > 5]),
            len([w for w in weights if w > self.avg(weights)]),
            len([w for w in weights if w > onepercent]),
            onepercent,
            KEY[self.source],
            len(sessions.keys()),
            KEY[self.source]))

        self.log('{} Total transitions (sum of all weights)'.format(sum([e[2]['weight'] for e in G.edges(data=True)])))
        self.log('{} Total transitions weight > 1 (sum of all weights > 1)'.format(sum([e[2]['weight'] for e in G.edges(data=True) if e[2]['weight'] > 1])))

        return G

    def getNodeName(self,state):
        name = None
        tmp = str(state[self.type])

        if self.type == 'ontology':
            try:
                name = int(tmp)
                name = None
            except:
                if tmp.lower() != 'nan' and len(tmp) > 0 and tmp != 'http':
                    name = tmp

        elif self.type == 'concept':
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
                                name = s[0]
                                self.log('len==1: {}: {} : {} <<<<<<----------------------------- this must be an error?'.format(self.type,name,state[self.type]))
                            else:
                                name = s[1]
                        else:
                            name = tmp

                elif tmp.lower() != 'nan':
                    name = tmp
        return name

    def readServerLogData(self):
        ### loading all data
        df = pd.read_csv(self.fn,sep=FIELD_SEP,compression=COMPRESSION)
        self.log('Shape all data ({}): {}'.format(self.fn,df.shape))

        ### sorting by datetime and req_id
        df['timestamp'] = pd.to_datetime(df.timestamp)
        df = df.sort_values(by=['timestamp','req_id'])

        ### Excluding some rows which DO NOT fulfil the condition
        if self.filterin is not None:
            for column_name, value in self.filterin.items():
                df = df.loc[df[column_name] == value]
        self.log('Shape filtered-in data ({}): {}'.format(self.fn,df.shape))

        ### Excluding some rows which DO fulfil the condition
        if self.filterout is not None:
            for column_name, value in self.filterout.items():
                df = df.loc[df[column_name] != value]
        self.log('Shape filtered-out data ({}): {}'.format(self.fn,df.shape))

        return df

    def getTransitionGraph(self):
        fn = self.getGraphFilename()
        if not self.exists(fn):
            return self.createTransitionGraph()
        else:
            self.log('Loading Transitions Graph: {}'.format(fn))
            return self.loadGraph(self.getGraphFilename())

    def removeDummies(self, G):
        self.log('Session graph original: {} edges'.format(G.number_of_edges()))
        ### Removing edges from and to START and INIT (dummy states)
        for dummy in DUMMIES:
            G.remove_node(dummy)
        self.log('Session graph without dummy states: {} edges'.format(G.number_of_edges()))
        return G

    def removeEdgesByWeight(self, G, threshold):
        ### Remove edges with weight == threshold
        for edge in G.edges(data=True):
            if edge[2]['weight'] == threshold:
                G.remove_edge(*edge[:2])
        self.log('Session graph without edges weight=1: {} edges'.format(G.number_of_edges()))
        return G

    #############################################################################
    # SESSIONS
    #############################################################################
    def createSessions(self,df):
        sessionsizes = []
        ipaddresses = set()

        if df is None:
            df = self.readLogData()

        if not self.sessionsExist():
            sessions = {}
            grouped = df.groupby([KEY[self.source]])

            for name, group in grouped:

                ### Clickstream: IP address | API Request: Api Key
                key = name
                session_seq = 0

                if key not in sessions:
                    sessions[key] = {session_seq:[]}

                ### CREATING SESSIONS (ALL CONSECUTIVE ACTIONS ---LESS THAN 30MIN BETWEEN EACH OTHER--- BELONG TO 1 SESSION)
                previous_timestamp = None
                for index, row in group.iterrows():

                    if not self.isConsecutive(row['timestamp'],previous_timestamp):
                        session_seq += 1
                        sessions[key][session_seq] = []

                    ipaddresses.add(row['ip'])

                    sessions[key][session_seq].append({'timestamp':row['timestamp'],
                                   'req_id':row['req_id'],
                                   'ip':row['ip'],
                                   'action':row['action'],
                                   'request':row['request'],
                                   'request_action':row['request_action'],
                                   'statuscode':row['statuscode'],
                                   'size':row['size'],
                                   'referer':row['referer'] if 'referer' in row else None,          #clickstream
                                   'useragent':row['useragent'] if 'useragent' in row else None,    #clickstream
                                   'user':row['user'] if 'user' in row else None,                   #apirequests
                                   'apikey':row['apikey'] if 'apikey' in row else None,             #apirequests
                                   'ontology':row['ontology'],
                                   'concept':row['concept']})
                    previous_timestamp = row['timestamp']

                tmpsize = [len(actions) for sequenceid,actions in sessions[key].items()]
                sessionsizes.extend(tmpsize)

            self.saveSessions(sessions)
        else:
            sessions = self.getSessions()
            sessionsizes = [len(actions) for key,obj in sessions.items() for sequenceid,actions in obj.items()]
            ipaddresses = set([action['ip'] for key,obj in sessions.items() for sequenceid,actions in obj.items() for action in actions])

        self.log('\n=== SUMMARY SESSIONS {} === \n# {}s({}) - {} TOTAL SESSIONS - SESSION_LENGHT_STATS( {} avg | {} median | {} mode | {} max | {} min | {} == 1 | {} > 1 | {} > avg | {} == max | {} IPs )'.format(
            self.name,
            self.source.upper(),
            len(sessions.keys()),
            sum([1 for key,obj in sessions.items() for seqid,actions in obj.items()]),
            self.avg(sessionsizes),
            self.median(sessionsizes),
            self.mode(sessionsizes),
            max(sessionsizes),
            min(sessionsizes),
            len([1 for ss in sessionsizes if ss == 1]),
            len([1 for ss in sessionsizes if ss > 1]),
            len([1 for ss in sessionsizes if ss > self.avg(sessionsizes)]),
            len([1 for ss in sessionsizes if ss == max(sessionsizes)]),
            len(ipaddresses)
        ))

        return sessions

    def saveSessions(self,sessions):
        try:
            self.savePickle(sessions,self.getSessionsFilename())
        except:
            return False
        return True

    def getSessions(self):
        return self.loadPickle(self.getSessionsFilename())

    def sessionsExist(self):
        return self.exists(self.getSessionsFilename())

    def isConsecutive(self,current_timestamp, previous_timestamp):
        if previous_timestamp is None:
            return True
        delta = self.deltaTimestamps(current_timestamp,previous_timestamp,TSFORMAT)
        return delta.total_seconds() < INNACTIVITY

    #############################################################################
    # PATH POSITION VS ...
    #############################################################################

    def getRelativePosition(self, index, sessionlength):
        return float('{0:.1f}'.format(index/float(sessionlength)))

    def _plotPositionVsAttribute(self,attribute,data,positions,attributes,fn):
        ### HEATMAP
        y = [positions.index(position) for position,obj in data.items() for attribut,counts in obj.items() if counts > 1]
        x = [attributes.index(attribut) for position,obj in data.items() for attribut,counts in obj.items() if counts > 1]
        z = [counts for position,obj in data.items() for attribut,counts in obj.items() if counts > 1]
        m = csr_matrix((z, (x, y)))
        self.plot_matrix(m,fn,
                         xtitle = 'Relative Path Position',
                         ytitle = '{} between transitions'.format(attribute.capitalize()),
                         bartitle = 'Counts')
        self.log('Positions: {}'.format(positions))

    ### Path Positon vs Duration
    def getCountsPositionVsDuration(self, session_length):
        sessions = self.getSessions()
        data = {}
        positions = set()
        durations = set()

        for key, obj in sessions.items():
            for sequenceid, actions in obj.items():

                timestamp1 = None
                nactions = len(actions)

                if ((session_length < MAX_LENGTH and nactions == session_length) or (session_length == MAX_LENGTH and nactions >= session_length)):

                    for index,action in enumerate(actions):

                        if timestamp1 is None:
                            timestamp1 = action['timestamp']
                        else:
                            timestamp2 = action['timestamp']
                            duration = round(self.deltaTimestamps(timestamp1,timestamp2,TSFORMAT).total_seconds())
                            position = self.getRelativePosition(index,nactions-1)

                            positions.add(position)
                            durations.add(duration)

                            if position not in data:
                                data[position] = {}
                            if duration not in data[position]:
                                data[position][duration] = 0
                            data[position][duration] += 1

                            timestamp1 = timestamp2

        return data,sorted(list(positions)),sorted(list(durations))

    def plotPositionVsDuration(self):
        session_lengths = [2,3,4,5,6] #number of nodes in session
        for session_length in session_lengths:
            self.log('=== SESION_LENGTH: {} nodes in session ==='.format(session_length))
            data,positions,attributes = self.getCountsPositionVsDuration(session_length)

            fn = self.getCountsPositionVsDurationPlotFileName(session_length)
            self._plotPositionVsAttribute('duration',data,positions,attributes,fn)

    ### Path Positon vs Search
    def getCountsPositionVsSearch(self):
        return None,None,None

    def plotPositionVsSearch(self):
        data,positions,attributes = self.getCountsPositionVsSearch()
        self._plotPositionVsAttribute('search',data,positions,attributes,None)

    #############################################################################
    # ENTRY AND EXIT POINTS
    #############################################################################
    def plotEntryAndExitPoints(self):

        fnentry = self.getEntryPointsDataFileName()
        fnexit = self.getExitPointsDataFileName()
        topk = 10

        if self.exists(fnentry) and self.exists(fnexit):
            entry_points = self.loadPickle(fnentry)
            exit_points = self.loadPickle(fnexit)
            self.log('Entry and Exit points Loaded!')
        else:
            threshold = 10
            G = self.loadGraph(self.getGraphFilename())

            while True:
                entry_points = {n.split('/')[-1]:G[DUMMYSTART][n]['weight'] for n in nx.all_neighbors(G,DUMMYSTART) if G[DUMMYSTART][n]['weight'] >= threshold}
                exit_points = {n.split('/')[-1]:G[n][DUMMYEND]['weight'] for n in nx.all_neighbors(G,DUMMYEND) if G[n][DUMMYEND]['weight'] >= threshold}
                if len(entry_points.keys()) >= 1 and len(exit_points.keys()) >= 1:
                    break
                elif threshold == 1:
                    self.log('THERE IS NO ENOUGH DATA TO PLOT.')
                else:
                    threshold = 1
            ### Saving data
            self.savePickle(entry_points,fnentry)
            self.savePickle(exit_points,fnexit)
            self.log('Entry and Exit points Saved!')

        ### TOPK
        entry_sorted = self.sortDict(entry_points,True,True)[:topk]
        exit_sorted = self.sortDict(exit_points,True,True)[:topk]
        entry_points = {nw[0]:nw[1] for nw in entry_sorted}
        exit_points = {nw[0]:nw[1] for nw in exit_sorted}

        ### DATAFRAME
        actions = list(set(entry_points.keys()).union(set(exit_points.keys())))
        data = {'action':actions, 'as_entry':[0 if n not in entry_points else entry_points[n] for n in actions], 'as_exit':[0 if n not in exit_points else exit_points[n] for n in actions]}
        df = pd.DataFrame(data, columns = data.keys())

        ### PLOT
        fn = self.getEntryAndExitPointsPlotFileName()
        self.plotGroupBars(df=df,
                           title='{}'.format('ALL ONTOLOGIES' if self.filterin is None else '{} CONCEPTS'.format(self.filterin['ontology'])),
                           xlabel=self.type.title(),
                           fn=fn,
                           labelkey='action',
                           groups=['as_entry','as_exit'],
                           ylog=True,
                           xticks=True,
                           posdelta=True)
        return

    def plotDistributionEdgeWeights(self):
        G = self.loadGraph(self.getGraphFilename())
        data = [edge[2]['weight'] for edge in  G.edges(data=True)]

        weights = list(set(data))
        counts = [data.count(w) for w in weights]

        fn = self.getDistributionEdgeWeightsPlotFileName()
        self.scatterPlot({'x':weights, 'y':counts},fn,
                         logy=True,logx=True,alpha=0.2,
                         xlabel='Edge-Weights',ylabel='Edge-Counts\n(frequency)',
                         title='{}'.format('ONTOLOGIES' if self.filterin is None else '{} CONCEPTS'.format(self.filterin['ontology'])))



    #############################################################################
    # TESTS
    #############################################################################

    def check_state_identifiers(self):
        sessions = self.getSessions()
        self.log('=== 0003 ===')
        self.log('ACTION | REQUEST | REQUEST ACTION | CONCEPT | REFERER | ONTOLOGY')
        for ip,obj in sessions.items():
            for seq,actions in obj.items():
                for action in actions:
                    tmp = str(action['concept'])
                    if 'http' in tmp:
                        tmp = urllib.unquote(urllib.unquote(tmp))
                        key = '/ontology/'
                        if key in tmp:
                            s = tmp[tmp.index(key)+len(key):].split('/')
                            if len(s) == 1:
                                self.log(': {} ({})'.format(s[0],tmp))
                            else:
                                self.log('- {} ({})'.format(s[1],tmp))
                        else:
                            self.log(str(action))
                            self.log('x {} ({})'.format(action['concept'],tmp))
                    elif tmp.lower() != 'nan':
                        self.log('* {}'.format(action['concept']))
        return

#############################################################################
# Main
# e.g.: python transitionParser.py clickstream /home/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'DRON'}" 0
#       python transitionParser.py clickstream /home/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ ontology None 0
#############################################################################
if __name__ == '__main__':
    source = utils.getParameter(1)      # clickstream, apirequests
    fn = utils.getParameter(2)          # data filelog
    results = utils.getParameter(3)     #pointing to results
    type = utils.getParameter(4)        # ontology (across ontologies), concept (within ontologies)
    filterin = utils.getParameter(5)    # "{'ontology':'<ontology_name>'}" or "{'ontologies':['<ontology_name>','<ontology_name>']}" or None
    filterout = utils.getParameter(6)   # "{'request_action':'Browse Ontology Class Tree'}" or None
    sizepath = utils.getParameter(7)    # integer (number of hops between source and target nodes)

    cp = TransitionParser(source,fn,results,filterin,filterout,type,sizepath)
    cp.loginit()
    cp.createTransitionGraph()
    # #cp.plotPositionVsDuration()
    # #cp.plotPositionVsSearch()
    cp.plotDistributionEdgeWeights()
    cp.plotEntryAndExitPoints()
    cp.logend()

    # ### CHECK SORTING
    # df = pd.read_csv(fn,sep=FIELD_SEP,compression=COMPRESSION)
    # df['timestamp'] = pd.to_datetime(df.timestamp)
    # df = df.sort_values(by='timestamp')
    # cp = Parser(fn,results,filter,type,sizepath)
    # counter = 0
    # print df.shape
    # for i,row in df.iterrows():
    #     print(i)
    #     if counter == 0:
    #         timestamp1 = row['timestamp']
    #         print('timestamp1: {}'.format(timestamp1))
    #     elif counter == 1:
    #         timestamp2 = row['timestamp']
    #         print('timestamp2: {}'.format(timestamp2))
    #     elif counter == 2:
    #         tmp = cp.deltaTimestamps(timestamp1, timestamp2, TSFORMAT)
    #         cp.log(tmp)
    #         print(str(tmp))
    #     else:
    #         break
    #     counter += 1
    # print 'end'
    # sys.exit(0)

    # ### CHECK CONCEPT NAMES FROM DATAFRAME
    # df = pd.read_csv(fn,sep=FIELD_SEP,compression=COMPRESSION)
    # df = df.loc[df['ontology'] == filter['ontology']]
    # df = df.loc[df['concept'] == 'http']
    # for index, row in df.iterrows():
    #     print row['concept'],row['request']
    # print df.shape
    # sys.exit(0)

    # ### CHECK ONTOLOGY NAMES FROM DATAFRAME
    # df = pd.read_csv(fn,sep=FIELD_SEP,compression=COMPRESSION)
    # grouped = df.groupby(['ontology'])
    # counter = 0
    #
    # for name, group in grouped:
    #
    #     try:
    #         id = int(name)
    #         counter += 1
    #         print id
    #         #for index, row in group.iterrows():
    #         #    print row
    #     except:
    #         continue
    #
    # print '{} numeric ids'.format(counter)
    # print df.shape
    # sys.exit(0)

    # ### CHECK CONCEPT NAMES FROM SESSIONS
    # cp = Parser(fn,results,filter,type)
    # cp.check_state_identifiers()
