__author__ = 'lisette.espin'

import utils
import sys
from joblib import Parallel, delayed
import multiprocessing
import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()

import networkx as nx
from collections import defaultdict
import json
import subprocess
import traceback
import numpy as np
import urllib

MUST = ['opt']
SUCCESS = 1
NOTHING = 0
ERR_COLS = -1
ERR_OTHER = -2
ERR_CSV = -2

###########################################################################
# METADATA
###########################################################################
def validate_data(df, params):
    metadataroot = params['ondb']
    n_jobs = multiprocessing.cpu_count() - 1
    results = Parallel(n_jobs=n_jobs)(delayed(read_data)(metadataroot, row['ontology'],row['year'],str(row['submissionId'])) for index,row in df.iterrows())
    df = pd.concat(results, ignore_index=True)
    print(df.head(10))
    return df

def read_data(metadataroot,ontology,year,submissionId):
    root = os.path.join(metadataroot,ontology,submissionId)

    if not os.path.exists(root):
        return None

    # ontology formats:
    # .csv.gz
    # .ttl
    # .obo
    # .owl

    for ext in ['.csv.gz','.ttl','.obo','.owl','.owl.zip','.rdf','.owl?format=raw','.obo?format=raw','.obo\?view=co']:
        fn = [x for x in os.listdir(root) if x.endswith(ext)]

        if len(fn) == 0:
            continue

        if os.path.exists(os.path.join(root,fn[0])):
            return pd.DataFrame({'ontology':[ontology],'year':[year],'ext':[ext]})

    print('No file available: {}-{}-{}: ({})'.format(ontology, year, submissionId, ','.join(os.listdir(root))))
    return pd.DataFrame({'ontology':[ontology],'year':[year],'ext':['other']})

def plot_validation_data(df, output):
    fn = os.path.join(output, 'ontology_validation_metadata.png')
    df['available'] = [1] * df.shape[0]
    g = df.groupby(['year','ext'])['available'].sum()
    print(g)
    axes = g.unstack(level=0).plot(kind='bar', subplots=True,  sharex=True, sharey=True, rot=45, ylim=(0,160), title='Available metadata per year (by format)')
    for ax in axes:
        ax.set(ylabel="# ontologies")
    axes[-1].set(xlabel="format")

    fig = plt.gcf()
    fig.savefig(fn, bbox_inches="tight")
    print('figure {} done!'.format(fn))
    plt.close()


###########################################################################
# GRAPHS
###########################################################################

def generate_graphs(df, params):

    # read ontology revisons (revisionpath), get revision id
    # load ontology by revsion_id (ontopath)
    # generate network
    # store edge list? adj. matrix? networkx graph? etc.

    n_jobs = multiprocessing.cpu_count() - 1
    results = Parallel(n_jobs=n_jobs)(delayed(generate_graph)(params['ondb'], row['ontology'], row['year'], str(row['submissionId']), params['on']) for index, row in df.iterrows())

    details = pd.DataFrame(columns=['ontology','year','status'])
    for r in results:

        if r[0] is None and r[1] is None and r[2] is None:
            continue

        status = 'SUCCESS' if r[2] == SUCCESS else 'NOTHING' if r[2] == NOTHING else 'ERR_COLS' if r[2] == ERR_COLS else 'ERR_CSV' if r[2] == ERR_CSV else 'OTHER'

        details = details.append(pd.DataFrame({'ontology':[r[0]],'year':[r[1]],'status':[status]}, columns=['ontology','year','status']), ignore_index=True)

    fn = os.path.join(params['on'],'summary_graph_generation.csv')
    details.to_csv(fn)

    return details

def generate_graph(ontopath, ontoname, year, submissionId, output):
    root = os.path.join(ontopath, ontoname, submissionId)

    if not os.path.exists(root):
        return (None,None,None)

    # ontology formats:
    # .csv.gz
    # .ttl
    # .obo
    # .owl

    for ext in ['.csv.gz', '.obo', '.owl', '.owl.zip', '.owl?format=raw', '.obo?format=raw', '.obo\?view=co', '.ttl', '.rdf' ]:
        fn = [x for x in os.listdir(root) if x.endswith(ext)]

        if len(fn) == 0:
            continue

        fn = os.path.join(root, fn[0])
        if os.path.exists(fn):

            if ext == '.csv.gz':
                return (ontoname,year,from_csv(ontoname,year,fn,output))

            # elif ext in ['.obo','.owl','.ttl','.owl?format=raw','.obo?format=raw','.obo\?view=co']:
            #     return from_parse(ontoname,year,fn,output)

            else:
                return (ontoname,year,NOTHING)

    return (ontoname,year,NOTHING)

def get_conceptid(concept, ontology):
    # tmp = concept.split('/')[-1].split('%2F')[-1].split('&')[0].split('%3A')[-1].split(':')[-1].split('#')[-1].split('_')[-1]
    # tmp = tmp[1:] if tmp.startswith('#') else tmp
    # tmp = urllib.parse.unquote(tmp).upper()

    tmp = urllib.parse.unquote(concept) #.split('conceptid=')[-1])
    tmp = tmp.split('/')[0].split('&')[0] if tmp.find('http') < 0 else tmp.split('/')[-1].split('&')[0]
    tmp = tmp[1:] if tmp.startswith('#') else tmp
    tmp = tmp.split(':')[-1].split('_')[-1].split('#')[-1] if tmp.lower().startswith(ontology.lower()) else tmp
    tmp = tmp.upper()

    if np.random.randint(0, 1000000, 1)[0] == 10:
        print('{}: {} --> {}'.format(ontology,concept, tmp))

    return tmp

def from_csv(ontoname,year,fnsource,output):

    try:
        df = pd.read_csv(fnsource, sep=',', compression='gzip')
    except:
        traceback.print_exc()
        return ERR_CSV

    try:
        columns = ['Class ID', 'Parents', 'Preferred Label', 'Synonyms', 'Semantic Types']
        if len(set(df.columns).intersection(set(columns))) == len(columns):
            G = nx.DiGraph(name=ontoname)
            metadata = defaultdict(defaultdict)

            # every node (concept in ontology)
            for id, row in df.iterrows():
                classid = row['Class ID']
                nodei = get_conceptid(classid,ontoname) # http://purl.bioontology.org/ontology/CPT/28805

                if row['Parents'] is not np.nan and row['Parents'] is not None and str(row['Parents']).lower() != 'nan':
                    # every father of concept
                    for nodej in row['Parents'].split('|'):
                        nodej = get_conceptid(nodej,ontoname)
                        G.add_edge(nodej, nodei)

                else:
                    G.add_node(nodei)

                # metadata
                metadata[nodei]['Class ID'] = classid
                if 'Preferred Label' in row and row['Preferred Label'] is not np.nan and row['Preferred Label'] is not None and str(row['Preferred Label']).lower() != 'nan':
                    s = row['Preferred Label']
                else:
                    s = ''
                metadata[nodei]['Preferred Label'] = s

                if 'Synonyms' in row and row['Synonyms'] is not np.nan and row['Synonyms'] is not None and str(row['Synonyms']).lower() != 'nan':
                    s = row['Synonyms'].split('|')
                else:
                    s = ''
                metadata[nodei]['Synonyms'] = s

                if 'Semantic Types' in row and row['Semantic Types'] is not np.nan and row['Semantic Types'] is not None and str(row['Semantic Types']).lower() != 'nan':
                    s = row['Semantic Types'].split('|')
                else:
                    s = ''
                metadata[nodei]['Semantic Types'] = s

            fn = os.path.join(output, 'graph', '{}_{}.adjlist'.format(ontoname.upper(), year))
            nx.write_adjlist(G, fn)

            fn = os.path.join(output, 'metadata', '{}_{}_metadata.json'.format(ontoname, year))
            with open(fn,'w') as f:
                json.dump(metadata,f)

            return SUCCESS
        else:
            return ERR_COLS

    except Exception:
        traceback.print_exc()
        return ERR_OTHER


def from_parse(ontoname, year, fnsource, output):

    # parsing
    ext = fnsource.split('.')[-1]
    newfn = os.path.join(output, '{}_{}_parsed{}.tmp'.format(ontoname, year,ext))
    try:
        # java -jar hierarchy-extractor.jar $1 $2
        args = ['java', '-jar', 'hierarchy-extractor.jar', fnsource, newfn]
        print('exec: {}'.format(' '.join(args)))
        r = subprocess.check_call(args)
    except Exception as ex:
        print(ex.message)
        return ERR_OTHER

    # creating graph
    if r == 0:
        G = nx.DiGraph(name=ontoname)
        with open(newfn, 'r') as f:
            for line in f:
                tmp = line.split('    ')

                child = tmp[0].replace('<', '').replace('>', '').strip('\n')
                parent = tmp[1].replace('<', '').replace('>', '').strip('\n')

                G.add_edge(parent, child)

        fn = os.path.join(output, 'graph', '{}_{}.adjlist'.format(ontoname, year))
        nx.write_adjlist(G, fn)

        # no metadata :-(

    return

def generate_ontology_graphs(params):
    fn = os.path.join(params['on'], 'summary_graph_generation.csv')
    if os.path.exists(fn):
        details = pd.read_csv(fn)
    else:
        df = summary(params)
        details = generate_graphs(df, params)

    plot_counts_grouped(details, xaxis='status', group='year', output=params['on'])
    plot_counts(details[details['status'] == 'SUCCESS'], 'year', params['on'])
    plot_counts(details[details['status'] == 'SUCCESS'], 'ontology', params['on'])
    plot_year_counts(details[details['status'] == 'SUCCESS'], params['on'])

    for threshold in [1, 2, 3, 4]:
        tmp = details[details['status'] == 'SUCCESS'].groupby('ontology')['year'].nunique()
        tmp = tmp[tmp == threshold]
        print('\n{} years ({} ontos): \n{}'.format(threshold, tmp.shape, tmp))

###########################################################################
# SUMMARY
###########################################################################
def summary(params):
    ontologies = _read_ontologies_parallel(params['rp'])
    df = pd.concat(ontologies, ignore_index=True)
    print(df.head(10))
    return df

def _read_ontologies_parallel(revisionpath):
    n_jobs = multiprocessing.cpu_count() -1
    fn = 'ontology_<onto>_revisions_by_year.json'
    ontologies = [x for x in os.listdir(revisionpath) if os.path.isdir(os.path.join(revisionpath,x)) and os.path.exists(os.path.join(revisionpath,x,fn.replace('<onto>',x)))]
    total_folders = [x for x in os.listdir(revisionpath) if os.path.isdir(os.path.join(revisionpath,x))]
    print('{} ontology folders out of {} | {} n_jobs'.format(len(ontologies),len(total_folders),n_jobs))
    return Parallel(n_jobs=n_jobs)(delayed(read_ontology)(revisionpath,ontoname) for ontoname in ontologies)

def read_ontology(revisionpath,ontoname):
    #{"2011": {"released": "2011-11-18T18:04:56-08:00", "creationDate": "2011-11-18T18:04:56-08:00", "version": "alpha 0.1", "submissionId": 11},
    # "2013": {"released": "2013-09-12T18:10:52-07:00", "creationDate": "2013-09-12T18:10:52-07:00", "version": "unknown", "submissionId": 45},
    # "2012": {"released": "2012-12-18T18:12:34-08:00", "creationDate": "2012-12-18T18:12:34-08:00", "version": "unknown", "submissionId": 32}}
    fn = 'ontology_{}_revisions_by_year.json'.format(ontoname)
    fn = os.path.join(revisionpath,ontoname,fn)
    cols = ['ontology','year','submissionId']
    if not os.path.exists(fn):
        print('{} does NOT exist.'.format(fn))
        return None
    else:
        with open(fn,'r') as f:
            data = json.load(f)
            rows = []
            for year in data.keys():
                rows.append(pd.DataFrame({'ontology':[ontoname],'year':[year],'submissionId':[data[year]['submissionId']]},columns=cols))
        if len(rows) == 0:
            print('no rows. {}'.format(ontoname))
            return None
        df = pd.concat(rows,ignore_index = True)
        return df

def plot_ontology_year(df,output):
    # ontology vs year
    fn = os.path.join(output, 'ontology_vs_year.png')
    df['available'] = [1] * df.shape[0]
    table = pd.pivot_table(df, values='available', index=['ontology'], columns = ['year'], fill_value=0)
    ax = sns.heatmap(table,cmap='Blues',yticklabels=False)
    ax.set_title('Ontology vs. Year')
    fig = ax.get_figure()
    fig.savefig(fn,bbox_inches="tight")
    print('figure {} done!'.format(fn))
    plt.close()

def plot_year_counts(df,output):
    # plot per year how many ontologies
    fn = os.path.join(output, 'ontology_year_counts.png')
    years = df.groupby(['year'])['ontology'].count()
    ax = years.plot(kind='bar')
    ax.set_title('Ontologies per Year')
    fig = ax.get_figure()
    fig.savefig(fn,bbox_inches="tight")
    print('figure {} done!'.format(fn))
    plt.close()

    return df

def plot_counts(df,xaxis, output):
    fn = os.path.join(output, 'counts_{}.png'.format(xaxis))
    g = sns.factorplot(x=xaxis, data=df, kind="count", size=4, aspect=.7)
    g.savefig(fn, bbox_inches="tight")
    print('figure {} done!'.format(fn))
    plt.close()

def plot_counts_grouped(df,xaxis,group,output):
    fn = os.path.join(output, 'counts_{}_per_{}.png'.format(xaxis,group))
    g = sns.factorplot(x=xaxis, col=group, data=df, kind="count", size=4, aspect=.7)
    g.savefig(fn, bbox_inches="tight")
    print('figure {} done!'.format(fn))
    plt.close()

###########################################################################
# MAIN
###########################################################################

if __name__ == '__main__':
    params = utils.get_params(MUST)

    if params['opt'] == 'filtering':
        utils.validate_params(params,['rp','on'])
        df = summary(params)
        plot_ontology_year(df,params['on'])
        plot_year_counts(df,params['on'])

    elif params['opt'] == 'dataval':
        utils.validate_params(params,['ondb','rp','on'])
        df = summary(params)
        df = validate_data(df,params)
        plot_validation_data(df, params['on'])

    elif params['opt'] == 'graph':
        utils.validate_params(params, ['ondb', 'rp', 'on'])
        generate_ontology_graphs(params)

# example
# nice -n10 python ontologies.py -opt filtering -rp bioportal/ontologies_revisions_per_year/ -on bioportal/ontologies/
# nice -n10 python ontologies.py -opt dataval -rp bioportal/ontologies_revisions_per_year/ -on bioportal/ontologies/ -ondb datasets/bioportal/ontologies/
# nice -n10 python ontologies.py -opt graph -rp bioportal/ontologies_revisions_per_year/ -on bioportal/ontologies/ -ondb datasets/bioportal/ontologies/