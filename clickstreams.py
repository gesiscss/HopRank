__author__ = 'lisette.espin'

import utils
import sys
from joblib import Parallel, delayed
import multiprocessing
import os
import json
import six
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# from matplotlib import cm
import networkx as nx
import datetime
import time
import urllib
from collections import Counter
from collections import OrderedDict
import networkx as nx
from collections import OrderedDict
import itertools
import progressbar
from time import sleep
import operator
from cycler import cycler
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy import io
import traceback
import gc

import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()

MUST = ['opt']
FIELD_SEP = ','
COMPRESSION = 'bz2'
TSFORMAT = '%Y-%m-%d %H:%M:%S'
ENCODING = 'ISO-8859-1'
SEARCH_ENGINES = ['google','bing','yahoo','baidu','aol','ask','duckduckgo','dogpile','excite','wolframalpha','yandex','lycos','chacha']
THRESHOLD_DEGREE = 0.1 # <0.1 flat, >=0.1 hierarchical
BREAK = 30 * 60 # 30 min
TSFORMAT = '%Y-%m-%d %H:%M:%S'
MINREQ = 2
LOGFILE = '/bigdata/lespin/bioportal/logs/log_<opt>_<date>.txt'
NAVITYPES = {'ALL':'ALL', 'DB':'direct_browsing', 'LS':'local_search', 'ES':'external_search', 'EL':'external_link', 'O':'other'}
TMPFOLDER = '/bigdata/lespin/tmp/'

###########################################################################
# HANDLERS
###########################################################################

def printf(str):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    msg = '{}\t{}'.format(st,str)
    print(msg)
    _log(msg)

def _log(msg):
    with open(LOGFILE, 'a') as f:
        f.write(msg)
        f.write('\n')

def save_df(df, fn):
    if os.path.exists(fn):
        printf('file {} already exists.'.format(fn))
    else:
        df.to_csv(fn)
        printf('file {} saved!'.format(fn))

def read_csv(fn):
    df = pd.read_csv(fn,index_col=0)

    if 'timestamp' in df:
        df['timestamp'] = pd.to_datetime(df.timestamp)

    printf('file {} loaded!'.format(fn))
    return df

def getlinestyles():
    linestyles = OrderedDict(
        [('solid', (0, ())),
         ('loosely dotted', (0, (1, 10))),
         ('dotted', (0, (1, 5))),
         ('densely dotted', (0, (1, 1))),

         ('loosely dashed', (0, (5, 10))),
         ('dashed', (0, (5, 5))),
         ('densely dashed', (0, (5, 1))),

         ('loosely dashdotted', (0, (3, 10, 1, 10))),
         ('dashdotted', (0, (3, 5, 1, 5))),
         ('densely dashdotted', (0, (3, 1, 1, 1))),

         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    linestyles = [(name, linestyle) for i, (name, linestyle) in linestyles.items()]
    linestyles = itertools.cycle(linestyles)
    return linestyles

class OrderedCounter(Counter, OrderedDict):
    pass

def find_ngrams(input_list, n):
    # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    return zip(*[input_list[i:] for i in range(n)])

def to_undirected(graph,weighted=False):
    H = nx.Graph()
    H.add_nodes_from(graph.nodes())
    if weighted:
        H.add_edges_from(graph.edges_iter(), weight=0)
        for u, v, d in graph.edges_iter(data=True):
            H[u][v]['weight'] += d['weight']
    else:
        H.add_edges_from(graph.edges_iter())
    return H

def load_sparse_matrix(fn):
    try:
        obj = io.mmread(fn)
        printf('file {} loaded!'.format(fn))
        return obj.tocsr()
    except Exception as ex:
        traceback.print_exc()
        printf(ex.message)
        return None

def save_sparse_matrix(m, fn):

    try:
        io.mmwrite(fn, m)
        printf('file {} saved!'.format(fn))
    except Exception as ex:
        printf(ex.message)
        return False
    return True

def slice_rows_cols(m,indices):
    m = m.tocsc()[:,indices]
    m = m.tocsr()[indices,:]
    return m

def logheader(params):
    printf(':::')
    printf('******************************************************')
    printf('Parameters:')
    for k,v in params.items():
        printf('{}\t{}'.format(k,v))
    printf('******************************************************')


###########################################################################
# PLOTS
###########################################################################

def render_mpl_table(data, prefix, col_width=0.6, row_height=0.5, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0.12, 0, 1, 1], header_columns=0,
                     ax=None, output=None, **kwargs):

    data = data.applymap('{:10,.0f}'.format)

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([6, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, rowLabels=data.index, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)

        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])


    if output is not None:
        ax.set_title('Clickstreams Filtering Summary')
        fn = os.path.join(output, '{}_clickstreams_summary.png'.format(prefix))
        plt.savefig(fn)
        plt.close()

def plot_clickstream_year(df, var, output, prefix, stacked=False):
    # ontology vs year
    # timestamp, req_id, ip, action, request, request_action, statuscode, size, referer, useragent ontology concept

    fn = os.path.join(output, '{}_clickstreams_{}{}_vs_year.png'.format(prefix, var, '-stacked' if stacked else ''))
    if os.path.exists(fn):
        printf('plot {} already exists.'.format(fn))
        return

    if var not in df.columns:
        printf('{} not in columns:{}'.format(var, df.columns))
        sys.exit(2)

    if not stacked:
        table = df.groupby(['year'])[var].nunique()
        ax = table.plot(kind='bar', logy=False)
    else:
        # timestamp
        table = df.groupby(['year', 'request_action'])['year'].count().unstack('request_action').fillna(0)
        ax = table.plot(kind='bar', stacked=True, logy=False, colormap='tab10')
        ax.get_legend().set_bbox_to_anchor((0.8, 1))

    ax.set_title('{} vs. Year'.format(var.title()))
    fig = ax.get_figure()
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def plot_requests_by_year(df, xaxis, output, prefix):
    fn = os.path.join(output, '{}_clickstream_year_{}_counts.png'.format(prefix,xaxis))

    g = df.groupby(['year', xaxis])
    printf(g)
    printf('-')

    g = g.filter(lambda x: len(x) > 1)
    printf(g)
    printf('--')

    fp = sns.factorplot(xaxis,
                        col='year',
                        data=g[['year', xaxis]],
                        kind='count',
                        col_wrap=2,
                        sharex=True,
                        sharey=True)

    for ax in fp.axes:
        ax.set(ylabel="# requests")
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yscale("log", nonposy='clip')
        ax.set(xlabel=xaxis)

    fp.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def plot_cdf(df, xaxis, output, prefix, norm=False):

    fn = os.path.join(output, '{}_clickstream_year_{}_cumsum{}.png'.format(prefix, xaxis, '_norm' if norm else ''))
    if os.path.exists(fn):
        printf('plot {} already exists.'.format(fn))
        return

    # requests per year
    for name, group in df.groupby(['year'])['year']:
        df.loc[df['year'] == name, 'total_req'] = group.count()

    # requests per year and ontology
    g = df.groupby(['year', 'total_req', xaxis]).count()
    g = g['req_id'].groupby(level=0, group_keys=False)
    res = g.apply(lambda x: x.sort_values(ascending=False))  # big to small

    # cumsum (grouped data)
    data = res.groupby(level=[0]).cumsum()

    # cumsum (table)
    tmp = data.reset_index(level=[0, 1, 2])
    tmp = tmp.rename(index=str, columns={"req_id": "cumsum"})

    # cumsum normalized
    if norm:
        tmp['cumsum'] = tmp['cumsum'] / tmp['total_req']

    # plots
    years = tmp['year'].unique()

    unit = 2.5
    fig = plt.figure(figsize=(unit * len(years), unit))
    counter = 0
    for i, year in enumerate(years):
        counter += 1;
        plt.subplot(1, len(years), counter)
        fp = sns.pointplot(x=xaxis,
                           col='year',
                           data=tmp[tmp['year'] == year],
                           y='cumsum')

        # top90
        if norm:
            x = 60 if prefix != 'validated' else 4 if year == 2013 else 10  # int(round(tmp['ontology'].nunique() / 4)) # 60
            y = 0.7
            c1 = tmp['year'] == year
            for top in [1.0, 0.99, 0.98, 0.95, 0.90, 0.80, 0.50]:
                plt.text(x, y, r'Top{}% req.: {} ontos.'.format(int(top * 100), tmp[c1 & (tmp['cumsum'] <= top)].shape[0]), size=8)
                y -= 0.1

        # hide xtick labels
        labels = [item.get_text() for item in fp.get_xticklabels()]
        empty_string_labels = [''] * len(labels)
        fp.set_xticklabels(empty_string_labels)

        # y-limits
        fp.set_ylim(0, 1 if norm else tmp['cumsum'].max())

        # set title (year)
        fp.set_title(year)

        # hide y-label
        if i > 0:
            fp.set_ylabel('')
        else:
            fp.set_ylabel('cumsum requests')

        # show ticks
        fp.tick_params(axis=u'both', which=u'both', length=3)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

    plt.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()
    df = df.drop(['total_req'], axis=1)

def plot_concepts_distribution(params):
    fn = os.path.join(params['cs'], 'valid_clickstreams.csv')

    if not os.path.exists(fn):
        printf('ERROR: {} does not exist.'.format(fn))
        sys.exit(2)

    df = read_csv(fn)
    for year in df['year'].unique():
        # data per year
        g = pd.DataFrame({'concepts': df.loc[df['year'] == year, ['_ontology', '_concept']].groupby(["_ontology"])['_concept'].nunique(),
                          'requests': df.loc[df['year'] == year, ['_ontology', 'req_id']].groupby(["_ontology"])['req_id'].count()}).reset_index()

        g = g.sort_values('concepts', ascending=False)

        # plot concepts per ontology
        _plot_ontology_year_by(g, year, params, x='_ontology', y='concepts')

        # plot requests per ontology
        _plot_ontology_year_by(g, year, params, x='_ontology', y='requests')

        # plot concepts vs requests
        ax = sns.jointplot(x="concepts", y="requests", data=g, kind="reg")
        fn = os.path.join(params['cs'], 'valid_clickstreams_ontology_concepts_vs_requests{}.png'.format(year))
        ax.savefig(fn, bbox_inches="tight")
        printf('figure {} done!'.format(fn))
        plt.close()

def _plot_ontology_year_by(g, year, params, x, y):
    ax = sns.barplot(x=x, y=y, data=g, log=g[y].max() >= 100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=6)
    ax.set_title('{} {}'.format(y, year))

    # number of ontologies having top# of concepts
    tmp = pd.DataFrame({'counts': g.groupby(y)[y].count()}).reset_index()
    for top in [10, 100, 1000]:
        if g[y].max() >= top:
            counts = tmp.loc[tmp[y] >= top, :]['counts'].sum()
            plt.text(counts, top, r'>={}: {} ontos.'.format(top, counts), size=10)

    # saving fig
    fn = os.path.join(params['cs'], 'valid_clickstreams_ontology_{}_{}.png'.format(y, year))
    ax.figure.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def plot_degree_distributions(params):

    df = validation(params,False)

    linestyles = getlinestyles()

    for year in df['year'].unique():
        tmp = df.loc[df['year']==year]

        fig, axes = plt.subplots(1,2,sharex=True,sharey=True)
        for ontology in tmp['_ontology'].unique():

            fn = os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(ontology, year))
            if not os.path.exists(fn):
                return None
            G = nx.read_adjlist(fn)
            k = G.number_of_edges() / G.number_of_nodes()

            axid = int(k >= THRESHOLD_DEGREE)
            degree_sequence = sorted([d for n, d in G.degree().items()], reverse=True)
            degreeCount = OrderedCounter(degree_sequence)
            deg, cnt = zip(*degreeCount.items())

            axes[axid].plot(deg, cnt, linestyle=next(linestyles),linewidth=1.5, label='{}, {:.2f}'.format(ontology,k)) #linewidth=1.5
            axes[axid].set_ylabel("Count")
            axes[axid].set_xlabel("Degree")

        axes[0].set_yscale("log")
        axes[0].set_xscale("log")

        axes[1].set_yscale("log")
        axes[1].set_xscale("log")

        axes[0].set_title('Flat (k < {})'.format(THRESHOLD_DEGREE))
        axes[1].set_title('Hierarchical (k >= {})'.format(THRESHOLD_DEGREE))

        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=False, ncol=2, prop={'size': 6})
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=False, ncol=2, prop={'size': 6})

        plt.suptitle("Degree Histogram {}".format(year))

        fn = os.path.join(params['cs'], 'valid_clickstreams_degree_distribution_{}.png'.format(year))
        plt.savefig(fn, bbox_inches="tight")
        plt.close()
        printf('file {} saved!'.format(fn))

def plot_session_navitypes(params):
    if 'topk' in params:
        try:
            topk = int(params['topk'])
        except Exception as e:
            printf('error topk not int: {}'.format(params['topk']))
            printf(e.message)
            sys.exit(2)

        prefix = 'top{}'.format(topk)

    else:
        prefix = 'valid'

    fn = os.path.join(params['cs'], '{}_clickstream_with_metadatas.csv'.format(prefix))

    if os.path.exists(fn):
        df = read_csv(fn)
    else:
        df = valid_metadata(params)

    _plot_navigation_types(df, params, prefix)

def _plot_navigation_types(df, params, prefix):
    norm = 'norm' in params and params['norm'].lower() in ['true', 'yes', 't', 'y']
    fs = 10 if norm and 'topk' in params else 6
    df.loc[df['navitype'] == 'direct_browsing', 'navitype'] = 'DB'
    df.loc[df['navitype'] == 'local_search', 'navitype'] = 'LS'
    df.loc[df['navitype'] == 'external_search', 'navitype'] = 'ES'
    df.loc[df['navitype'] == 'external_link', 'navitype'] = 'EL'
    df.loc[df['navitype'] == 'other', 'navitype'] = 'O'

    for year in df['year'].unique():
        fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, gridspec_kw={'width_ratios': [1, 2 if fs == 6 else 5]})
        _fig, _ax = plt.subplots(1, 1)

        table = df.loc[df['year'] == year, :].groupby(['_ontology', 'navitype'])['year'].count().unstack('navitype').fillna(0)
        table.plot(ax=_ax, kind='bar', stacked=True)

        for id, dt in enumerate(['flat', 'hierarchical']):
            # plot ontology vs. requests stacked as navitype
            table = df.loc[(df['year'] == year) & (df['degreetype'] == dt), :].groupby(['_ontology', 'navitype'])['year'].count().unstack('navitype').fillna(0)

            if norm:
                table = table.apply(lambda x: x / x.sum(), axis=1)

            table.plot(ax=axes[id], kind='bar', stacked=True)  # , colormap='tab10')

            axes[id].get_legend().remove()
            axes[id].set_xlabel('')

            axes[id].set_xticklabels(axes[id].get_xticklabels(), rotation=75, size=fs)
            axes[id].set_title(dt.title())

        handles, labels = _ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', fancybox=True, shadow=False, ncol=5, prop={'size': 10})  # , bbox_to_anchor=(0.5, -0.15)
        # plt.suptitle('# Requests per navitype and _ontology {}'.format(year))

        fn = os.path.join(params['cs'], '{}_clickstreams__ontology_navitype_{}{}.png'.format(prefix,
                                                                                             year,
                                                                                             '_norm' if norm else ''))
        fig.savefig(fn, bbox_inches="tight")
        printf('figure {} done!'.format(fn))
        plt.close()

def _plot_time(df, scale, prefix, params, postfix=None):
    if scale not in ['day', 'weekday', 'hour', 'weekhour']:
        printf('{} is not implemented.'.format(scale))
        sys.exit(2)

    df['date'] = df['timestamp'].dt.date if scale == 'day' else df['timestamp'].dt.weekday if scale == 'weekday' else df['timestamp'].dt.hour if scale == 'hour' else (df[
                                                                                                                                                                           'timestamp'].dt.weekday * 24) + \
                                                                                                                                                                      df[
                                                                                                                                                                          'timestamp'].dt.hour if scale == 'weekhour' else None
    tmp = df.drop_duplicates(['date', 'ip', 'sessionseq'])
    tmp = tmp.groupby(['date', 'ip', 'sessionseq']).size().reset_index().rename(columns={0: 'count'})
    tmp = tmp.groupby('date')['count'].sum()
    tmp = pd.DataFrame(tmp, columns=['count']).reset_index(level=0)
    ax = tmp.plot(x='date', y='count', rot=45)
    ax.set_ylabel('# sessions')
    ax.set_xlabel(scale)
    if scale == 'weekday':
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('# Sessions per {} {}'.format(scale, params['year']))
    fig = ax.get_figure()
    fn = os.path.join(params['cs'], 'sessions_per_{}_{}_{}{}{}.png'.format(scale, params['year'], prefix, '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                           '_{}'.format(postfix) if postfix is not None else ''))
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def _plot_user(df, prefix, params, postfix=None):
    # sessions per user
    df['date'] = df['timestamp'].dt.year
    tmp = df.drop_duplicates(['date', 'ip', 'sessionseq'])
    tmp = tmp.groupby(['date', 'ip', 'sessionseq']).size().reset_index().rename(columns={0: 'count'})
    tmp = tmp.groupby('ip')['count'].sum()
    tmp = pd.DataFrame(tmp, columns=['count']).sort_values('count')
    ax = tmp.plot(rot=45, logy=True)
    ax.set_ylabel('# sessions')
    plt.title('# Sessions per IP {}'.format(params['year']))
    fig = ax.get_figure()
    fn = os.path.join(params['cs'], 'sessions_per_IP_{}_{}{}{}.png'.format(params['year'],
                                                                           prefix,
                                                                           '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                           '_{}'.format(postfix) if postfix is not None else ''))
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def _plot_sessions_per_ontology(df, prefix, params, postfix=None):
    var = '_ontology'
    tmp = df.drop_duplicates(['_ontology', 'ip', 'sessionseq'])
    tmp = tmp.groupby([var, 'ip', 'sessionseq']).size().reset_index().rename(columns={0: 'count'})
    tmp = tmp.groupby(var)['count'].sum()
    tmp = pd.DataFrame(tmp, columns=['count']).sort_values('count')
    ax = tmp.plot(rot=90, logy=True)

    ax.set_ylabel('# sessions')
    fig = ax.get_figure()
    plt.title('# sessions per {} {}'.format(var, params['year']))

    if var == '_ontology':
        ontos = pd.DataFrame(tmp, columns=['count']).sort_values('count').reset_index()
        top99 = ['CHMO', 'NDDF', 'RXNORM', 'VANDF', 'COSTART', 'CPT', 'CRISP', 'HL7', 'ICD10', 'ICD9CM', 'ICPC', 'LOINC', 'MEDDRA', 'MEDLINEPLUS', 'MESH', 'NDFRT', 'OMIM', 'RCD', 'SNMI',
                 'SNOMEDCT', 'TAO', 'WHO-ART']
        ontos = ontos.loc[ontos[var].isin(top99), :]
        plt.xticks(ontos.index, ontos._ontology, rotation=90)

    fn = os.path.join(params['cs'], 'sessions_per_{}_{}_{}{}{}.png'.format(var,
                                                                           params['year'],
                                                                           prefix, '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                           '_{}'.format(postfix) if postfix is not None else ''))
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def _plot_clicks_per_sessionid(df, prefix, params, postfix=None):
    tmp = df.drop_duplicates(['req_id', 'ip', 'sessionseq'])
    tmp = tmp.groupby(['ip', 'sessionseq']).size().reset_index().rename(columns={0: 'count'})
    ax = tmp.sort_values('count').reset_index()['count'].plot(logy=True)

    ax.set_ylabel('# clicks')
    ax.set_xlabel('session')
    fig = ax.get_figure()
    plt.title('# clicks per session {}'.format(params['year']))

    fn = os.path.join(params['cs'], 'sessions_nclicks_{}_{}{}{}.png'.format(params['year'],
                                                                            prefix, '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                            '_{}'.format(postfix) if postfix is not None else ''))
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def _plot_mean_session_length_per_ontology(df, prefix, params, postfix=None):
    tmp = df.groupby(['_ontology', 'ip', 'sessionseq']).size().reset_index().rename(columns={0: 'mean'})
    tmp = tmp.groupby('_ontology')['mean'].mean()
    tmp = pd.DataFrame(tmp, columns=['mean']).sort_values('mean')
    ax = tmp.plot(rot=90, logy=True)

    ontos = pd.DataFrame(tmp, columns=['mean']).sort_values('mean').reset_index()
    top99 = ['CHMO', 'NDDF', 'RXNORM', 'VANDF', 'COSTART', 'CPT', 'CRISP', 'HL7', 'ICD10', 'ICD9CM', 'ICPC', 'LOINC', 'MEDDRA', 'MEDLINEPLUS', 'MESH', 'NDFRT', 'OMIM', 'RCD', 'SNMI', 'SNOMEDCT',
             'TAO', 'WHO-ART']
    ontos = ontos.loc[ontos._ontology.isin(top99), :]
    plt.xticks(ontos.index, ontos._ontology, rotation=90)
    fig = ax.get_figure()

    ax.set_ylabel('mean session length')
    fig = ax.get_figure()
    plt.title('session lengths {}'.format(params['year']))

    fn = os.path.join(params['cs'], 'sessions_mean_length_per_ontology_{}_{}{}{}.png'.format(params['year'],
                                                                                             prefix, '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                                             '_{}'.format(postfix) if postfix is not None else ''))
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def _plot_session_length_per_ontology(df, prefix, params, postfix=None):
    # top ontologies
    ontologies = df.drop_duplicates(['_ontology', 'ip', 'sessionseq'])
    ontologies = ontologies.groupby(['_ontology', 'ip', 'sessionseq']).size().reset_index().rename(columns={0: 'count'})
    ontologies = ontologies.groupby('_ontology')['count'].sum().sort_values(ascending=False).reset_index().rename(columns={0: 'count'})
    ontologies = ontologies.query('count > 1')  # more than 1 session pero onto

    # session lengths
    tmp = df.groupby(['_ontology', 'ip', 'sessionseq']).size().reset_index().rename(columns={0: 'count'})
    tmp = tmp.query('count > 1')  # more than 1 click per session

    # topk ontologies
    top = 5
    ontologies = ontologies.loc[ontologies._ontology.isin(tmp._ontology), :]
    ontologies = ontologies._ontology.values
    ontologies = np.append(ontologies[:top], ontologies[-top:])
    chunks = np.split(ontologies, 2)

    # colormap = plt.cm.spectral
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    linestyles = getlinestyles()

    for z, chunk in enumerate(chunks):
        # ax[z].set_prop_cycle(cycler('color',[colormap(i) for i in np.linspace(0, 0.9, top*2)]))

        for ontology in chunk:
            group = tmp.query("_ontology == '{}'".format(ontology))
            length_sequence = group.sort_values('count')['count'].values
            length_count = OrderedCounter(length_sequence)
            length, cnt = zip(*length_count.items())

            ax[z].plot(length, cnt, label=ontology, linewidth=1.5)  # , linestyle=next(linestyles)) #, linestyle=next(linestyles))
            ax[z].set_yscale("log")
            ax[z].set_xscale("log")

    ax[0].set_title("Top{}".format(top))
    ax[1].set_title("Bottom{}".format(top))

    ax[0].set_xlabel('session length')
    ax[1].set_xlabel('session length')
    ax[0].set_ylabel('count')

    plt.suptitle('Session Length Distribution {}'.format(2015))
    ax[0].legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 6})
    ax[1].legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 6})

    fn = os.path.join(params['cs'], 'sessions_length_dist_{}_{}{}{}.png'.format(params['year'], prefix, '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                                '_{}'.format(postfix) if postfix is not None else ''))
    fig.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()

def _plots(df, params, postfix=None):
    # sessions in time
    for scale in ['day', 'weekday', 'hour', 'weekhour']:
        _plot_time(df, scale, params['set'], params, postfix)

    # sessions per user
    _plot_user(df, params['set'], params, postfix)

    # sessions per ontology
    _plot_sessions_per_ontology(df, params['set'], params, postfix)

    # clicks per session
    _plot_clicks_per_sessionid(df, params['set'], params, postfix)

    # session length per ontology
    _plot_mean_session_length_per_ontology(df, params['set'], params, postfix)
    _plot_session_length_per_ontology(df, params['set'], params, postfix)

    # actions per user

    # ontologies per user

def plot_sessions(params):
    df = generate_session_ids(params)
    _plots(df, params)

def plot_transitions(params):
    df = transitions(params)
    _plots(df, params, 'final')
    _plot_navigation_types(df, params, 'final')
    _plot_possition_in_session(df, params)

def _plot_possition_in_session(df, params):

    # sns.set(style="whitegrid")
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}
    sns.set_context("paper", rc=paper_rc)
    sns.set(font_scale=1.5)

    norm = 'norm' in params and params['norm'].lower() in ['y', 't']
    ontos = ['SNOMEDCT', 'MEDDRA', 'CPT', 'RXNORM', 'NDDF', 'CHMO']
    tmp = df.loc[df._ontology.isin(ontos)].groupby(['_ontology', 'navitype', 'position']).size().reset_index().rename(columns={0: 'count'})
    types = np.sort(tmp.navitype.unique())

    mi = tmp['position'].min()
    ma = tmp['position'].max()
    mc = tmp['count'].max()
    # printf(mi,ma,mc)

    if norm:
        for i, v in tmp.groupby(['_ontology', 'position'])['count'].sum().iteritems():
            tmp.loc[(tmp._ontology == i[0]) & (tmp.position == i[1]), 'count'] /= float(v)

    g = sns.factorplot(x="position", y="count", hue="navitype", row="_ontology", data=tmp, sharex=True, sharey=True, size=1., aspect=7.5, row_order=ontos, hue_order=types, legend=False)

    for ax, onto in zip(g.axes, ontos):
        ax = ax[0]
        ax.set_title('')
        ax.set_ylabel(onto)
        plt.setp(ax.collections, sizes=[1])
        if not norm:
            ax.set(xscale="log", yscale="log")
            ax.set_ylim(1, mc)
        else:
            ax.set(xscale="log")
            ax.set_ylim(0, 1)
        ax.set_xlim(1, ma + 1)

    plt.subplots_adjust(top=1.20, bottom=0.1, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    plt.legend(bbox_to_anchor=(0.90, 1.25), markerscale=5.0, loc='upper right', borderaxespad=0., bbox_transform=plt.gcf().transFigure, ncol=len(types))

    fn = os.path.join(params['cs'], 'transitions_top{}_position_in_session_{}{}{}.png'.format(params['topk'],
                                                                                              params['year'],
                                                                                              '_withinonto' if 'withinonto' in params and params['withinonto'].lower()[0] in ['y', 't'] else '',
                                                                                              '_norm' if norm else ''))

    g.savefig(fn, bbox_inches="tight")
    printf('figure {} done!'.format(fn))
    plt.close()


###########################################################################
# VALID DATA (CROSS VALIDATION)
###########################################################################

def current_summary(df, title):
    printf('{}:'.format(title))
    printf('{} years'.format(df['year'].unique().shape))
    printf('{} ontologies'.format(df['ontology'].unique().shape))
    printf('{} concepts'.format(df['concept'].unique().shape))
    if '_ontologies' in df.columns and '_concepts' in df.columns:
        printf('{} _ontologies'.format(df['_ontology'].unique().shape))
        printf('{} _concepts'.format(df['_concept'].unique().shape))
    printf('{} records'.format(df.shape))
    printf('')

def _get_valid_data(params):

    # load filtered data
    fn = os.path.join(params['cs'], 'valid_clickstreams.csv')

    if os.path.exists(fn):
        df = read_csv(fn)
        return df

    return None

def validation(params, plots=True):

    # get filtered data
    df = _get_valid_data(params)

    if df is None:
        df = filtering(params, False)
        current_summary(df,'Validating')

        # add condition that ontology should exist
        printf('Condition 1...')
        df = validate_ontologies(df, params)
        current_summary(df, 'Condition1 (ontology exists)')

        # add condition that concept should exist
        printf('Condition 2...')
        df = validate_concepts(df, params)
        current_summary(df, 'Condition2 (concept exists)')

        printf('Checking...')
        c3 = len(df.loc[(df._ontology.isnull()) | (df._ontology is np.nan),:])
        printf('{} _ontologies null or nan.'.format(c3))
        c4 = len(df.loc[(df._concept.isnull()) | (df._concept is np.nan), :])
        printf('{} _concept null or nan.'.format(c4))
        if c3 > 0 or c4 > 0:
            printf('WARNING: c3:{} >0 or c4:{} >0'.format(c3,c4))
            sys.exit(0)

        if plots:
            # save
            fn = os.path.join(params['cs'], 'valid_clickstreams.csv')
            df.to_csv(fn)

        if plots:
            # plot statistics after validation
            plot_clickstream_year(df, 'ontology', params['cs'], 'validated')
            plot_clickstream_year(df, '_ontology', params['cs'], 'validated')
            plot_clickstream_year(df, 'concept', params['cs'], 'validated')
            plot_clickstream_year(df, '_concept', params['cs'], 'validated')
            plot_clickstream_year(df, 'ip', params['cs'], 'validated')
            plot_clickstream_year(df, 'request', params['cs'], 'validated')
            plot_clickstream_year(df, 'request_action', params['cs'], 'validated')
            plot_clickstream_year(df, 'request_action', params['cs'], 'validated', True)
            plot_cdf(df, 'ontology', params['cs'], 'validated', norm=True)
            plot_cdf(df, 'ontology', params['cs'], 'validated', norm=False)
            plot_cdf(df, '_ontology', params['cs'], 'validated', norm=True)
            plot_cdf(df, '_ontology', params['cs'], 'validated', norm=False)

        # summary
        fn = os.path.join(params['cs'], 'filtered_clickstreams_summary.csv')
        dfsummary = pd.read_csv(fn)

        df.loc[:,'available'] = 1
        tmp = df.groupby(['year'])['available'].agg(['count']).transpose()
        tmp.columns.name = ''
        tmp.columns = tmp.columns.astype(str)
        tmp.reset_index(drop=True)
        tmp['index'] = '-CrossVal'
        tmp.index.name = ''
        columns = [tmp.columns[-1]]
        columns.extend(tmp.columns[0:-1])
        tmp = tmp[columns]
        df = df.drop(['available'], axis=1)

        dfsummary.columns = dfsummary.columns.astype(str)
        dfsummary = dfsummary.append(tmp, ignore_index=True)
        dfsummary.set_index('index', inplace=True, verify_integrity=True)
        printf(dfsummary)

        if plots:
            fn = os.path.join(params['cs'], 'valid_clickstreams_summary.csv')
            dfsummary.to_csv(fn)
            printf('file {} saved!'.format(fn))
            render_mpl_table(dfsummary, prefix='valid', header_columns=0, output=params['cs'])

    return df

def validate_ontologies(df, params):
    fn = os.path.join(params['cs'], 'filtered_clickstreams_crossval_ontologies.csv')
    if os.path.exists(fn):
        printf('loading file: {}...'.format(fn))
        return read_csv(fn)

    groups = [name for name, group in df.groupby(['year', '_ontology'])]
    total = len(groups)
    printf('{} groups (year-ontology)'.format(total))

    df.loc[:, 'remove'] = 0

    bar = progressbar.ProgressBar(maxval=total, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i = 0
    for year, ontology in groups:
        i += 1

        fnyo = os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(ontology, year))
        df.loc[(df['year'] == year) & (df['_ontology'] == ontology), 'remove'] = not os.path.exists(fnyo)
        bar.update(i)

    bar.finish()
    printf('end\n')

    tmp = df.loc[df['remove']==1,:]
    if len(tmp) > 0:
        printf('To remove {} out of {}: \n{}'.format(len(tmp), len(df), tmp[['year','request','ontology','_ontology']].sample(10 if len(tmp) > 10 else len(tmp))))
        tmp.to_csv(os.path.join(params['cs'], 'toremove', 'clickstreams_ontologies_to_remove.csv'))

    printf('removing {} records...'.format(df['remove'].sum()))
    df = df.loc[df['remove'] != 1,:]
    df = df.drop(['remove'], axis=1)
    df.to_csv(fn)
    del (tmp)
    printf('{} records after matching ontologies'.format(df.shape[0]))
    return df

def validate_concepts(df, params):
    fn = os.path.join(params['cs'], 'filtered_clickstreams_crossval_ontologies_crossval_concepts.csv')
    if os.path.exists(fn):
        printf('loading file: {}...'.format(fn))
        return read_csv(fn)

    groups = [name for name, group in df.groupby(['year', '_ontology'])]
    total = len(groups)
    printf('{} groups (year-ontology)'.format(total))
    printf('{} records before matching'.format(df.shape[0]))

    n_jobs = multiprocessing.cpu_count() - 1
    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs, temp_folder=TMPFOLDER)(delayed(_validate_concepts)(df.query("year == {} & ontology == '{}'".format(year,ontology)), year,ontology, params) for year, ontology in groups)
    printf('end')

    df = pd.concat([cs for cs in results if cs is not None], ignore_index=True)
    df.to_csv(fn)
    printf('{} records after matching concepts'.format(df.shape[0]))

    printf('file {} saved!'.format(fn))

    return df

def _validate_concepts(df, year, ontology, params):

    fn = os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(ontology, year))

    if not os.path.exists(fn):
        return None

    try:
        G = nx.read_adjlist(fn)
    except:
        printf('{}-{}:{} ERROR'.format(ontology,year,fn))
        return None

    df.loc[:, 'remove'] = 1
    df.loc[df['_concept'].isin(G.nodes()), 'remove'] = 0
    # df.loc[df['_concept'] == 'ROOT', 'remove'] = 0
    # df.loc[df['_concept'] == 'OWL#THING', 'remove'] = 0

    printf('To remove {} out of {}'.format(len(df.loc[df['remove'] == 1, :]), len(df)))
    df.loc[df['remove'] == 1, :].to_csv(os.path.join(params['cs'], 'toremove', 'clickstreams_concepts_to_remove_{}_{}.csv'.format(year, ontology)))
    df = df.loc[df['remove'] != 1,:]
    df = df.drop(['remove'], axis=1)
    del(G)

    return df


###########################################################################
# FILTERED DATA
###########################################################################

def init_analytics(params):

    # load data from scratch
    df = load_clickstreams(params)

    printf('========= ORIGINAL DATA =========')
    current_summary(df, 'Original')

    # plot statistics before filtering
    plot_clickstream_year(df, 'ontology', params['cs'], 'original')
    plot_clickstream_year(df, 'concept', params['cs'], 'original')
    plot_clickstream_year(df, 'ip', params['cs'], 'original')
    plot_clickstream_year(df, 'request', params['cs'], 'original')
    plot_clickstream_year(df, 'request_action', params['cs'], 'original')
    plot_clickstream_year(df, 'request_action', params['cs'], 'original', True)

    # cdf
    plot_cdf(df, 'ontology', params['cs'], 'original', norm=True)
    plot_cdf(df, 'ontology', params['cs'], 'original', norm=False)

    # storing dataset (original)
    fn = os.path.join(params['cs'], 'original_clickstreams.csv')
    save_df(df, fn)

    # filtering request_actions
    dra = summary_request_actions(df)
    fn = os.path.join(params['cs'], 'original_clickstreams_request_actions.csv')
    save_df(dra, fn)

def summary_request_actions(df):
    request_actions = ['Ontology Summary', 'Browse Ontology Classes', 'Browse Ontology Class', 'Browse Ontology Class Tree', 'Browse Ontology Mappings', 'Ontology Analytics',
                       'Browse Ontology Widgets', 'Browse Ontology Visualization', 'Browse Ontology Notes', 'Browse Ontology Properties', 'Browse Widgets', 'Browse Ontology Property Tree',
                       'Browse Class Notes']
    dra = pd.Series(index=request_actions)
    for ra in request_actions:
        c = df.loc[df['request_action'] == ra].shape[0]
        printf('{}: {} records.'.format(ra, c))
        dra.set_value(ra, c)
    return dra

def _get_filtered_data(params):

    # load filtered data
    fnfiltered = os.path.join(params['cs'], 'filtered_clickstreams.csv')
    filtered = False

    if os.path.exists(fnfiltered):
        df = read_csv(fnfiltered)
        filtered = True
    else:
        # load data from scratch
        df = load_clickstreams(params)

    printf(df.head(10))

    return df, filtered, fnfiltered

def filtering(params, plots=True):

    df, filtered, fnfiltered = _get_filtered_data(params)

    if not filtered:

        # filtering requests
        dfsummary, df = filters(df)

        # standarize ontology/concept column: only ID
        printf('Standarizing IDs...')
        df = preprocess_ontology_concept_ids(df)

        if plots:
            printf(df[['request','ontology','_ontology','concept','_concept']].sample(10))

        # storing files
        save_df(df, fnfiltered)
        fn = os.path.join(params['cs'], 'filtered_clickstreams_summary.csv')
        save_df(dfsummary, fn)
        if plots:
            render_mpl_table(dfsummary, prefix='filtered', header_columns=0, output=params['cs'])

    if plots:
        printf('========= FILTERED DATA =========')
        current_summary(df, 'Filtered')

        # plot statistics after filtering
        plot_clickstream_year(df, 'ontology', params['cs'], 'filtered')
        plot_clickstream_year(df, '_ontology', params['cs'], 'filtered')
        plot_clickstream_year(df, 'concept', params['cs'], 'filtered')
        plot_clickstream_year(df, '_concept', params['cs'], 'filtered')
        plot_clickstream_year(df, 'ip', params['cs'], 'filtered')
        plot_clickstream_year(df, 'request', params['cs'], 'filtered')
        plot_clickstream_year(df, 'request_action', params['cs'], 'filtered')
        plot_clickstream_year(df, 'request_action', params['cs'], 'filtered', True)

        # cdf
        plot_cdf(df, 'ontology', params['cs'], 'filtered', norm=True)
        plot_cdf(df, 'ontology', params['cs'], 'filtered', norm=False)
        plot_cdf(df, '_ontology', params['cs'], 'filtered', norm=True)
        plot_cdf(df, '_ontology', params['cs'], 'filtered', norm=False)

    # filtering request_actions
    if plots:
        dra = summary_request_actions(df)
        fn = os.path.join(params['cs'], 'filtered_clickstreams_request_actions.csv')
        save_df(dra, fn)

    return df

def filters(df):
    index = []
    printf('all records: {}'.format(df.shape[0]))

    df.loc[:,'available'] = 1
    dfsummary = df.groupby(['year'])['available'].agg(['count']).transpose()
    index.append('None')

    printf('Filtering...')

    criterion1 = df['request'].map(lambda x: '/ontologies/' in x)
    df = df[criterion1]
    dfsummary = dfsummary.append(df.groupby(['year'])['available'].agg(['count']).transpose(), ignore_index=True)
    index.append('+Ontology')
    printf('filter 1: {}'.format(df.shape[0]))

    criterion2 = df['request'].map(lambda x: 'p=classes' in x)
    df = df[criterion2]
    dfsummary = dfsummary.append(df.groupby(['year'])['available'].agg(['count']).transpose(), ignore_index=True)
    index.append('+Classes')
    printf('filter 2: {}'.format(df.shape[0]))

    criterion3 = df['request'].map(lambda x: 'conceptid=' in x)
    df = df[criterion3]
    dfsummary = dfsummary.append(df.groupby(['year'])['available'].agg(['count']).transpose(), ignore_index=True)
    index.append('+ConceptID')
    printf('filter 3: {}'.format(df.shape[0]))

    criterion4 = df['request'].map(lambda x: 'login' not in x and 'redirect' not in x and 'widgets' not in x and 'feedback' not in x and 'accounts' not in x and 'annotator' not in x and 'visualization' not in x and 'visualize' not in x and 'notes' not in x and 'mappings' not in x)
    df = df[criterion4]
    dfsummary = dfsummary.append(df.groupby(['year'])['available'].agg(['count']).transpose(), ignore_index=True)
    index.append('-ORequests')
    printf('filter 4: {}'.format(df.shape[0]))

    criterion5 = df['request'].map(lambda x: 'json_search' not in x and 'callback=jQuery' not in x and 'callback=children' not in x)
    df = df[criterion5]
    dfsummary = dfsummary.append(df.groupby(['year'])['available'].agg(['count']).transpose(), ignore_index=True)
    index.append('-AJAX')
    printf('filter 5: {}'.format(df.shape[0]))

    df = df.loc[df['statuscode'] == 200]
    dfsummary = dfsummary.append(df.groupby(['year'])['available'].agg(['count']).transpose(), ignore_index=True)
    index.append('+HTTP(200)')
    printf('filter 6: {}'.format(df.shape[0]))

    dfsummary['index'] = index
    dfsummary.set_index('index', inplace=True)

    df = df.drop(['available'], axis=1)
    printf(df.head(50))
    printf(dfsummary)
    return dfsummary, df

def preprocess_ontology_concept_ids(df):

    n_jobs = multiprocessing.cpu_count() - 1
    sizechunk = 100000
    nchunks = round(len(df) / sizechunk)

    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs)(delayed(_preprocess_ontology_concept_ids)(g, dfchunk, nchunks) for g, dfchunk in df.groupby(np.arange(len(df)) // sizechunk))
    printf('end')

    printf('Recovering...')
    df = pd.concat([r for r in results if r is not None], ignore_index=True)
    printf('{} records'.format(len(df)))

    printf('Removing nans...')
    df = df.loc[(df._concept.notnull()) & (df._ontology.notnull()) & (df._concept is not np.nan) & (df._ontology is not np.nan), :]
    printf('{} records (after removing nans)'.format(len(df)))

    printf('Random Sample:')
    printf(df[['ontology', '_ontology', 'concept', '_concept']].sample(10))

    printf('Differences ontology and _ontology:')
    tmp = df.loc[(df.ontology.str.upper() != df._ontology.str.upper()),['request', 'ontology', '_ontology']]
    printf('None' if len(tmp)==0 else tmp.sample(10))
    printf(len(df))

    printf('Differences concept and _concept:')
    tmp = df.loc[(df.concept.str.upper() != df._concept.str.upper()), ['request', 'ontology','concept', '_concept']]
    printf('None' if len(tmp) == 0 else tmp.sample(10))
    printf(len(df))
    del(tmp)

    return df

def _preprocess_ontology_concept_ids(g, dfchunk, nchunks):

    printf('chunk#{} out of {}'.format(g,nchunks))
    # standarize ontology ID
    dfchunk.loc[:,'_ontology'] = dfchunk['request']
    dfchunk._ontology.replace(r"^.*?/ontologies/(.*?)($|[/\?].*$)", r"\1", regex=True, inplace=True)
    dfchunk.loc[:, '_ontology'] = dfchunk.apply(lambda r: r._ontology.upper(), axis=1)

    # standarize concept ID
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: urllib.parse.unquote(r.request.split('conceptid=')[-1]), axis=1)
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: r._concept.split('&')[0], axis=1)
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: r._concept if r._concept.find('/ontologies/') < 0 else r._concept.split('/')[-1], axis=1)
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: r._concept.split('/')[0] if r._concept.find('http') < 0 else r._concept.split('/')[-1], axis=1)
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: r._concept[1:] if r._concept.startswith('#') else r._concept, axis=1)
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: r._concept.split(':')[-1].split('_')[-1].split('#')[-1] if r._concept.lower().startswith(r.ontology.lower()) else r._concept, axis=1)
    dfchunk.loc[:, '_concept'] = dfchunk.apply(lambda r: r._concept.upper(), axis=1)

    return dfchunk

def load_clickstreams(params):
    clickstreams = read_clickstream_parallel(params['cslog'])
    df = pd.concat([cs for cs in clickstreams if cs is not None], ignore_index=True)
    return df

def read_clickstream_parallel(root):
    n_jobs = multiprocessing.cpu_count() -1
    clickstreams = [x for x in os.listdir(root) if os.path.isfile(os.path.join(root,x)) and x.endswith('.csv.bz2')]
    printf('{} clickstream files | {} n_jobs'.format(len(clickstreams),n_jobs))
    return Parallel(n_jobs=n_jobs)(delayed(read_clickstream)(os.path.join(root,csfn)) for csfn in clickstreams)

def read_clickstream(fn):
    # BP_webpage_requests_2013.csv.bz2
    # timestamp, req_id, ip, action, request, request_action, statuscode, size, referer, useragent ontology concept
    year = fn.split('.')[0].split('_')[-1]
    printf('year: {}'.format(year))

    if not os.path.exists(fn):
        printf('{} does NOT exist.'.format(fn))
        return None
    else:
        df = pd.read_csv(fn, sep=FIELD_SEP, compression=COMPRESSION, encoding = ENCODING, index_col=0)

        printf('Extracting year from timestamp...')

        if 'timestamp' in df:
            df['timestamp'] = pd.to_datetime(df.timestamp)
            df['year'] = pd.DatetimeIndex(df['timestamp']).year

        printf('Preprocessing (ontology and concept IDs): {}'.format(year))
        printf('{} years'.format(df['year'].unique().shape))
        printf('{} ontologies'.format(df['ontology'].unique().shape))
        printf('{} concepts'.format(df['concept'].unique().shape))
        printf('')

        return df


###########################################################################
# METADATA: navitype and degreetype
###########################################################################

def valid_metadata(params):

    fnnt = os.path.join(params['cs'], 'valid_clickstreams_with_metadata.csv')
    if os.path.exists(fnnt):
        df = read_csv(fnnt)
    else:
        df = validation(params, False)

        # navitypes
        df = _add_navi_types(df)

        # degree types
        df = _add_degree_types(df)

        # save
        df.to_csv(fnnt)

    return df

def _add_navi_types(df):

    if 'navitype' in df.columns:
        df.drop(['navitype'], axis=1, inplace=True)

    # local search
    df.loc[df['request'].str.contains('jump_to_nav=true'), 'navitype'] = 'local_search'
    printf('\n\nlocal search:')
    printf(df.groupby('navitype')['navitype'].count())

    # external search
    for se in SEARCH_ENGINES:
        df.loc[df['navitype'].isnull() & df['referer'].notnull() & df['referer'].str.contains(se), 'navitype'] = 'external_search'
    printf('\n\nexternal search:')
    printf(df.groupby('navitype')['navitype'].count())

    # direct browsing
    df.loc[df['navitype'].isnull() & df['referer'].notnull() & (df['referer'].str.startswith('http://bioportal.bioontology.org') | df['referer'].str.startswith('https://bioportal.bioontology.org')), 'navitype'] = 'direct_browsing'
    printf('\n\ndirect browsing:')
    printf(df.groupby('navitype')['navitype'].count())

    # external link
    df.loc[df['navitype'].isnull() & df['referer'].notnull() & df['referer'].str.startswith('http'), 'navitype'] = 'external_link'
    printf('\n\nexternal link:')
    printf(df.groupby('navitype')['navitype'].count())

    # others
    df.loc[df['navitype'].isnull(), 'navitype'] = 'other'
    printf('\n\nother:')
    printf(df.groupby('navitype')['navitype'].count())

    # final distribution
    printf('Navitype asssignment (to _ontologies)')
    printf(df.groupby('navitype')['_ontology'].nunique())
    return df

def _add_degree_types(df):

    if 'degreetype' in df.columns:
        df.drop(['degreetype'], axis=1, inplace=True)

    groups = [name for name, group in df.groupby(['year', '_ontology'])]
    total = len(groups)
    printf('{} groups (year-ontology)'.format(total))
    printf('{} records before matching'.format(df.shape[0]))

    n_jobs = multiprocessing.cpu_count() - 1
    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs)(delayed(_infer_degree_type)(df.loc[(df['year'] == year) & (df['_ontology'] == ontology)], year, ontology, params) for year, ontology in groups)
    printf('end')

    df = pd.concat([cs for cs in results if cs is not None], ignore_index=True)
    printf('\n\nflat vs. hierarchical (requests):')
    printf(df.groupby('degreetype')['degreetype'].count())

    printf('\n\nflat vs. hierarchical (unique _ontologies):')
    printf(df.groupby('degreetype')['_ontology'].nunique())

    return df

def _infer_degree_type(df, year, ontology, params):

    fn = os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(ontology, year))

    if not os.path.exists(fn):
        return None

    G = nx.read_adjlist(fn)
    k = G.number_of_edges() / G.number_of_nodes()

    degreetype = 'flat' if k < THRESHOLD_DEGREE else 'hierarchical'
    df['degreetype'] = degreetype

    return df

def top_requests(params):

    try:
        topk = int(params['topk'])
    except Exception as e:
        printf('error topk not int: {}'.format(params['topk']))
        printf(e.message)
        sys.exit(2)

    fn = os.path.join(params['cs'], 'top{}_valid_clickstreams_with_metadata.csv'.format(topk))
    if os.path.exists(fn):
        df = read_csv(fn)
    else:

        df = valid_metadata(params)
        key = 'top{}'.format(topk)
        df.loc[:,key] = 0

        # requests per year
        for name, group in df.groupby(['year'])['year']:
            df.loc[df['year'] == name, 'total_req'] = group.count()

        # requests per year and ontology
        g = df.groupby(['year','_ontology', 'total_req' ]).count()
        g = g['req_id'].groupby(level=0, group_keys=False)
        res = g.apply(lambda x: x.sort_values(ascending=False))  # big to small

        # cumsum (grouped data)
        data = res.groupby(level=[0]).cumsum()

        # cumsum (table)
        tmp = data.reset_index(level=[0, 1, 2])
        tmp = tmp.rename(index=str, columns={"req_id": "cumsum"})

        # cumsum normalized
        tmp['cumsum'] = tmp['cumsum'] / tmp['total_req']

        # years
        years = tmp['year'].unique()

        for i, year in enumerate(years):
            printf('{}...'.format(year))
            topdf = tmp.loc[(tmp['year']==year) & (tmp['cumsum']<=(topk/100.)),:]
            df.loc[(df['year']==year) & (df['_ontology'].isin(topdf['_ontology'])), key] = 1

        printf('done!')
        printf('\n{}\n'.format(df[['year','_ontology',key]].sample(10)))
        printf(df.groupby('year')['_ontology'].nunique())

        df = df.drop(['total_req'], axis=1)
        df = df.loc[df[key] == 1]

        df = df.drop([key], axis=1)
        df.to_csv(fn)

        printf('done (only {}!)'.format(key))
        printf('\n{}\n'.format(df[['year', '_ontology']].sample(10)))
        printf(df.groupby('year')['_ontology'].nunique())
        printf(df.shape)

    return df


###########################################################################
# SESSIONS
###########################################################################

def get_delta_time(timestamp1, timestamp2, tsformat=None):
    # t1 = datetime.datetime.strptime(str(timestamp1),tsformat)
    # t2 = datetime.datetime.strptime(str(timestamp2),tsformat)
    return abs(timestamp1 - timestamp2).total_seconds()

def generate_session_ids(params):

    if params['set'] not in ['filtered','valid']:
        printf('{} does not exist.'.format(params['set']))
        sys.exit(2)

    fn = os.path.join(params['cs'], 'sessions_clickstreams_with_metadata_{}_{}{}{}.csv'.format(params['year'],
                                                                                               params['set'],
                                                                                               '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                                               '_withinonto' if 'withinonto' in params and params['withinonto'].lower()[0] in ['y','t'] else ''))

    if os.path.exists(fn):
        return read_csv(fn)

    year = int(params['year'])
    printf('Generating session ids: {}'.format(year))
    df = filtering(params, plots=False)

    if params['set'] == 'valid':

        if 'topk' in params:
            dfvalid = top_requests(params)
        else:
            dfvalid = valid_metadata(params)

        df.loc[:,'valid'] = 0
        df.loc[df.req_id.isin(dfvalid.req_id),'valid'] = 1

        printf('Adding metadata...')
        df = df.join(dfvalid[['req_id','navitype','degreetype']].set_index('req_id'), on='req_id')
        printf(df.columns)
        printf('Sample: \n{}'.format(df.sample(5)))

    else:
        df.loc[:,'valid'] = 1

    printf('{} records (all years).'.format(len(df)))
    printf('=== {} ==='.format(params['set']))
    printf('valid==1: {}'.format(len(df.loc[df.valid == 1, :])))
    printf('valid==0: {}'.format(len(df.loc[df.valid == 0, :])))

    printf('1. selecting only year={}'.format(params['year']))
    df = df.query('year == {}'.format(year))
    printf('{} records.'.format(len(df)))

    printf('2. filtering out users (IPs) with less than {} requests.'.format(MINREQ))
    df = df.groupby(['ip'],as_index=False).filter(lambda x: len(x) >= MINREQ)
    printf('{} records.'.format(len(df)))

    printf('3. sortby timestamp and req_id')
    df.sort_values(by=['timestamp', 'req_id'], inplace=True)

    printf('4. groupby IP')
    dfg = df.groupby(['ip'], as_index=False)

    ### assigning session sequences to requests per IP (parallel)
    total = len(dfg)
    printf('{} total IPs'.format(total))
    printf('{} total records.'.format(len(df)))
    n_jobs = multiprocessing.cpu_count() - 1
    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs)(delayed(_generate_session_ids_parallel)(name, group, 'withinonto' in params and params['withinonto'].lower()[0] in ['y','t']) for name, group in dfg)
    printf('end')
    df = pd.concat(results, ignore_index=True)
    df = df.query('sessionseq > 0')
    printf('{} records after assigning seesionseq.'.format(len(df)))
    printf('5. filtering out sessions with less than {} requests.'.format(MINREQ))
    df = df.groupby(['ip','sessionseq'], as_index=False).filter(lambda x: len(x) >= MINREQ)
    printf('{} records after removing sessions with less than {}.'.format(len(df),MINREQ))
    ### assigning session sequences to requests per IP (parallel)

    for c in ['Unnamed: 0',  'Unnamed: 0.1']:
        if c in df.columns:
            df = df.drop(c,axis=1)

    save_df(df,fn)
    printf(df.sample(10))
    return df

def _generate_session_ids_parallel(name, group, withinontology=False):

    sessionseq = 1
    previous = None
    previous_onto = None

    group.sort_values('timestamp', inplace=True) # check this line (maybe redundant)

    for index, row in group.iterrows():

        if previous is None and row['valid'] == 1:
            previous = row['timestamp']
            previous_onto = row['_ontology']
            group.loc[index,'sessionseq'] = sessionseq
            continue
        elif previous is None and row['valid'] == 0:
            group.loc[index, 'sessionseq'] = -1
            continue

        current = row['timestamp']
        current_onto = row['_ontology']
        delta = get_delta_time(previous,current,TSFORMAT)

        if withinontology:
            if delta > BREAK or row['valid'] == 0 or previous_onto!=current_onto:
                sessionseq += 1
        else:
            if delta > BREAK or row['valid'] == 0:
                sessionseq += 1

        if row['valid'] == 1:
            group.loc[index, 'sessionseq'] = sessionseq
            previous = row['timestamp']
            previous_onto = row['_ontology']
        else:
            group.loc[index, 'sessionseq'] = -1
            previous = None
            previous_onto = None

    return group


###########################################################################
# TRANSITIONS
###########################################################################

def validate_navitype(params):
    if 'navitype' not in params:
        printf('missing arugment: navitype')
        sys.exit(2)

    params['navitype'] = params['navitype'].upper()
    if params['navitype'] not in NAVITYPES.keys():
        printf('navitype {} does not exist.'.format(params['navitype']))
        sys.exit(2)

def transitions(params):

    validate_navitype(params)

    year = params['year']
    fn = os.path.join(params['cs'], 'transitions_with_metadata_{}_{}{}{}_{}.csv'.format(year,
                                                                         params['set'],
                                                                         '_top{}'.format(params['topk']) if 'topk' in params else '',
                                                                         '_withinonto' if 'withinonto' in params and params['withinonto'].lower()[0] in ['y','t'] else '',
                                                                         params['navitype'].upper()))

    if os.path.exists(fn):
        df = read_csv(fn)
    else:
        df = generate_session_ids(params)

        n_jobs = multiprocessing.cpu_count() - 1
        printf('{} n_jobs'.format(n_jobs))
        results = Parallel(n_jobs=n_jobs, temp_folder=TMPFOLDER)(delayed(_create_transitions_graph)(df, ontology, params) for ontology in df._ontology.unique())
        printf('end')

        printf('Recovering...')
        df = pd.concat([r for r in results if r is not None], ignore_index=True)
        printf('{} records ({} out of {} ontologies.)'.format(len(df), len([r for r in results if r is not False]), len(results)))

        printf('Removing sessions with less than {} clicks...'.format(MINREQ))
        df = df.loc[df.remove != 1,:]
        printf('{} records after removal.'.format(len(df)))

        for c in ['remove','valid']:
            if c in df:
                df = df.drop(c, axis=1)

        printf('Adding possition in session to requests...')
        df = add_possition_in_session(df)

        df.to_csv(fn)
        printf('file {} saved!'.format(fn))

    return df

def _create_transitions_graph(df,ontology,params):

    target_navitype = NAVITYPES[params['navitype']]
    tmp = df.query("_ontology=='{}'".format(ontology))
    try:
        tmp = tmp.assign(remove=lambda x: 0)

        G = nx.DiGraph()
        for ipse, session in tmp.groupby(['ip', 'sessionseq']):

            concepts = session.sort_values('timestamp')['_concept'].unique()

            if len(concepts) >= MINREQ:
                actions = session.sort_values('timestamp')[['_concept', 'navitype']]
                action_list = [(r['_concept'], r['navitype']) for i, r in actions.iterrows()]

                for edge in list(find_ngrams(action_list, 2)):
                    if edge[0][0] != edge[1][0]:
                        if target_navitype == 'ALL' or edge[1][1] == target_navitype:
                            printf('target_navitype:{} \t edge:{} | accepted'.format(target_navitype, edge))

                            if not G.has_edge(edge[0][0], edge[1][0]):
                                G.add_edge(edge[0][0], edge[1][0], weight=0.0)
                            G[edge[0][0]][edge[1][0]]['weight'] += 1.0

                        else:
                            printf('target_navitype:{} \t edge:{} | ignored'.format(target_navitype, edge))
                    else:
                        printf('self-loop ({}) ignored.'.format(edge))

            else:
                tmp.loc[tmp.req_id.isin(session.req_id), 'remove'] = 1

        G.name = '{}-{}'.format(ontology,params['year'])
        fn = os.path.join(params['cs'], 'graph', params['navitype'], '{}_{}.adjlist'.format(ontology.upper(), params['year']))
        nx.write_weighted_edgelist(G, fn)

        printf('{}-{}: N({}), E({}), ME({}) done.'.format(ontology,params['year'],G.number_of_nodes(),G.size(),G.size(weight='weight')))
        return tmp

    except Exception as ex:
        printf('{} | {} | {} | {}'.format(ex,ontology, len(df), 'ERROR'))

    return None

def add_possition_in_session(df):

    n_jobs = multiprocessing.cpu_count() - 1
    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs, temp_folder=TMPFOLDER)(delayed(_possition_in_session)(df, ontology) for ontology in df._ontology.unique())
    printf('end')

    printf('Recovering...')
    df = pd.concat([r for r in results if r is not None], ignore_index=True)
    printf('{} records'.format(len(df)))

    return df

def _possition_in_session(df, ontology):

    tmp = df.query("_ontology == '{}'".format(ontology))
    tmp = tmp.assign(position=lambda x: 0)

    for name, group in tmp.groupby(['ip','sessionseq']):
        position = 1
        for i,row in group.sort_values('timestamp').iterrows():
            tmp.loc[tmp.req_id == row.req_id, 'position'] = position
            position += 1

    return tmp

def summary(params):
    df = transitions(params)
    year = params['year']

    columns = ['Ontology','Nodes','Edges','Average Degree','Sessions','% Visited Nodes','\# Visited Nodes', 'Unique Transitions','Multiple Transitions','Average Transitions','IPs']
    dfsummary = pd.DataFrame(columns=columns)

    for o in df._ontology.unique():
        Go = nx.read_adjlist(os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(o.upper(), year)))
        Gt = nx.read_weighted_edgelist(os.path.join(params['cs'], 'graph', NAVITYPES[params['navitype']],'{}_{}.adjlist'.format(o.upper(), year)))
        dfo = df.query("_ontology == '{}'".format(o))
        tmp = pd.DataFrame({'Ontology':[o],
                           'Nodes':[Go.number_of_nodes()],
                           'Edges':[Go.number_of_edges()],
                           'Average Degree':[round(Go.number_of_edges()/Go.number_of_nodes(),1)],
                           'Sessions':[dfo.groupby(['ip','sessionseq']).ngroups],
                           '% Visited Nodes':[int(round((Gt.number_of_nodes()*100)/Go.number_of_nodes(),0))],
                           '\# Visited Nodes': [Gt.number_of_nodes()],
                           'Unique Transitions':[int(Gt.size())],
                           'Multiple Transitions':[int(Gt.size(weight='weight'))],
                           'Average Transitions': [round(Gt.number_of_edges()/Gt.number_of_nodes(),1)],
                           'IPs':[dfo.ip.nunique()],
                           }, columns=columns)
        dfsummary = dfsummary.append(tmp, ignore_index=True)
        printf(tmp.iloc[0,:])

    dfsummary = dfsummary.sort_values(['Multiple Transitions'], ascending=False).reset_index(drop=True)
    dfsummary.index = dfsummary.index + 1
    printf('totals: \n{}'.format(dfsummary[['Sessions','Unique Transitions','Multiple Transitions','IPs']].sum()))
    fn = os.path.join(params['cs'],'summary_bioportal_{}.tex'.format(year))
    dfsummary.to_latex(fn)
    printf('file {} saved!'.format(fn))


###########################################################################
# HOPs
###########################################################################

def create_hops_adj(params):

    df = transitions(params)
    ontologies = df._ontology.unique()
    del(df)

    n_jobs = int(round(multiprocessing.cpu_count() / 4))
    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs, temp_folder=TMPFOLDER)(delayed(_visited_hops_ontology)(ontology, params['year'], params) for ontology in ontologies if ontology != 'CPT')
    printf('end')

    printf('SUMMARY:')
    for o,y,k in results:
        printf('{},{}: {} done!'.format(o,y,k))

def _visited_hops_ontology(ontology, year, params):

    fn_ontology = os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(ontology.upper(), year))
    Go = nx.read_adjlist(fn_ontology)
    # printf('file {} loaded!'.format(fn))
    Go = to_undirected(Go,False)
    nodes = sorted(Go.nodes())
    Ao = csr_matrix(nx.to_scipy_sparse_matrix(Go, nodelist=nodes, weight=None, format='csr'))
    # printf('graph to marix!')
    del(Go)

    fn_hop = os.path.join(params['on'],'hops','{}_{}_<k>HOP.mtx'.format(ontology, year))
    kdone = None
    for k in range(1,11,1):

        fn = fn_hop.replace('<k>',str(k))
        if os.path.exists(fn):
            printf('=== {}-{}: {}HOP already exists (pass)'.format(ontology, year, k))
            kdone = k
            continue

        printf('=== {}-{}: {}HOP ==='.format(ontology,year,k))
        if k == 1:
            hop = Ao.copy()
            shape = hop.shape
        else:
            hop = csr_matrix(hop.dot(Ao))
            # printf('1. dot product')

            hop = sparse.find(hop)
            # printf('2. >0')

            hop = csr_matrix((np.ones(hop[2].size).astype(np.int8), (hop[0], hop[1])), shape, dtype=np.int8)
            # printf('3. >0 --> 1')

            hop = hop.tolil()
            # printf('4. to lil')

            hop.setdiag(0)
            # printf('5. diagonal zero')

            hop = hop.tocsr()
            # printf('6. csr')

            hop.eliminate_zeros()
            # printf('6. eliminate zeros')

            if hop.sum() > 0:

                # removing previous HOPS
                for previous_k in range(k - 1, 0, -1):

                    previous_hop = load_sparse_matrix(fn_hop.replace('<k>',str(previous_k)))
                    # printf('9. loaded previous k done: {}'.format(previous_k))

                    hop = hop - previous_hop
                    # printf('10. minus')

                    hop = (hop > 0).astype(np.int8)
                    # printf('11. >0')

                    hop = hop.tolil()
                    # printf('12. to lil')

                    hop.setdiag(0)
                    # printf('13. diagonal to 0')

                    hop = hop.tocsr()
                    # printf('14. to csr')

                    hop.eliminate_zeros()
                    # printf('15. eliminate zeros')

                    if hop.sum() == 0:
                        printf('the matrix has already reached zero (break). Up to {}HOP'.format(previous_k))
                        break

            else:
                printf('the matrix has already reached zero (break). Up to {}HOP'.format(k-1))
                break

        printf('saving {}-{} {}hop...'.format(ontology,year,k))
        save_sparse_matrix(hop, fn)
        printf('{}-{} {}hop saved!'.format(ontology, year, k))
        kdone = k

    gc.collect()
    printf('=== {}-{}: done for {} HOPs! ==='.format(ontology, year, kdone))
    return (ontology,year,kdone)

def hops_overlap(params):

    validate_navitype(params)

    path_hops = os.path.join(params['on'], 'hops')
    onto_year_khops = [x.split('.')[0].replace('HOP','').split('_') for x in os.listdir(path_hops) if os.path.isfile(os.path.join(path_hops, x)) and x.endswith('HOP.mtx')]
    printf('{} files (ontology_year_kHOP) to be analyzed...'.format(len(onto_year_khops)))

    n_jobs = multiprocessing.cpu_count() - 1
    printf('{} n_jobs'.format(n_jobs))
    results = Parallel(n_jobs=n_jobs, temp_folder=TMPFOLDER)(delayed(_khop_overlap_ontology_year)(ontology, year, k, params) for ontology, year, k in onto_year_khops)
    printf('end')

    printf('SUMMARY:')
    for o,y,k,msg in results:
        printf('{},{}: {} {}!'.format(o,y,k,msg))

def _khop_overlap_ontology_year(ontology, year, k, params):

    fn_ontology = os.path.join(params['on'], 'graph', '{}_{}.adjlist'.format(ontology.upper(), year))
    fn_hop = os.path.join(params['on'], 'hops', '{}_{}_{}HOP.mtx'.format(ontology.upper(), year, k))
    fn_hop_overlap = os.path.join(params['cs'], 'hopsoverlap', NAVITYPES[params['navitype']], '{}_{}_{}HOP.mtx'.format(ontology.upper(), year, k))
    fn_hop_overlap_weighted = os.path.join(params['cs'], 'hopsoverlap', NAVITYPES[params['navitype']], '{}_{}_{}HOP_weighted.mtx'.format(ontology.upper(), year, k))
    fn_transitions = os.path.join(params['cs'], 'graph', NAVITYPES[params['navitype']],'{}_{}.adjlist'.format(ontology.upper(), year))

    if os.path.exists(fn_hop_overlap):
        printf('{}-{} {}hop: already exists.'.format(ontology,year,k))
        return (ontology, year, k, 'already existed.')

    printf('=== {}_{}: {}HOP, loading data...'.format(ontology,year,k))

    if not os.path.exists(fn_transitions):
        printf('file {} does not exist (skipt)'.format(fn_transitions))
        return (ontology, year, k, 'skipt')

    Go = nx.read_adjlist(fn_ontology)
    Gt = nx.read_weighted_edgelist(fn_transitions)

    printf('- {}_{}: {}HOP, only visited nodes...'.format(ontology, year, k))
    nodes_o = sorted(Go.nodes())
    nodes_t = sorted(Gt.nodes())
    printf('same concepts: {}'.format(len(set(nodes_o).intersection(set(nodes_t)))))
    del(Go)

    Ao = load_sparse_matrix(fn_hop)

    # unweighted
    At = nx.to_scipy_sparse_matrix(Gt, nodelist=nodes_t, weight=None)

    printf('- {}_{}: {}HOP, slicing adjacency matrices...'.format(ontology, year, k))
    nodes_o = np.array(nodes_o)
    nodes_t = np.array(nodes_t)
    indices = np.in1d(nodes_o, nodes_t, assume_unique=True)

    Ao = slice_rows_cols(Ao, indices)

    if Ao.shape != At.shape:
        printf('- {}_{}: {}HOP, shapes are different. {} != {}'.format(ontology, year, k, Ao.shape, At.shape))
        return (ontology, year, k, 'different shapes ({} != {})'.format(Ao.shape, At.shape))

    printf('- {}_{}: {}HOP, overlap unweighted...'.format(ontology, year, k))
    Aoverlap = Ao.multiply(At)

    printf('- {}_{}: {}HOP, saving...'.format(ontology, year, k))
    save_sparse_matrix(Aoverlap, fn_hop_overlap)

    # weighted:
    printf('- {}_{}: {}HOP, overlap weighted...'.format(ontology, year, k))
    At = nx.to_scipy_sparse_matrix(Gt, nodelist=nodes_t, weight='weighted')
    del (Gt)
    Aoverlap = Ao.multiply(At)
    printf('- {}_{}: {}HOP, saving...'.format(ontology, year, k))
    save_sparse_matrix(Aoverlap, fn_hop_overlap_weighted)

    printf('- {}_{}: {}HOP, done!...'.format(ontology, year, k))

    return (ontology,year,k,'done')

def validate_rel(params):
    if 'rel' not in params:
        printf('missing arugment: rel')
        sys.exit(2)

    params['rel'] = params['rel'].upper()
    if params['rel'] not in ['O','T','MT','RAW']:
        printf('reltype {} does not exist.'.format(params['navitype']))
        sys.exit(2)

def hops_overlap_summary(params):
    # index: #
    # columns: ontology, hop, navitype, overlap
    validate_rel(params)

    for navitype in NAVITYPES.keys():
        path_hop_overlaps = os.path.join(params['cs'], 'hopsoverlap', navitype)
        onto_year_khop = [x.split('.')[0].replace('HOP', '').split('_') for x in os.listdir(path_hop_overlaps) if os.path.isfile(os.path.join(path_hop_overlaps, x)) and x.endswith('HOP.mtx')]
        printf('{} files (ontology_year_kHOP) to be analyzed...'.format(len(onto_year_khop)))

        n_jobs = multiprocessing.cpu_count() - 1
        printf('{} n_jobs'.format(n_jobs))
        results = Parallel(n_jobs=n_jobs, temp_folder=TMPFOLDER)(delayed(_hop_overlap_summary)(navitype, ontology, year, k, params) for ontology, year, k in onto_year_khop)
        printf('end')

    printf('Reconstructing...')
    cols = ['ontology','hop','navitype','overlap']
    df = pd.DataFrame(columns=cols)

    for ontology,k,navitype,overlap in results:
        df = df.append({'ontology':ontology, 'hop':k, 'navitype':navitype, 'overlap':overlap}, columns=cols)

    printf(df.head(5))
    fn = os.path.join(params['cs','hopsoverlap','summary_rel{}_{}.csv'.format(params['rel'],params['year'])])
    save_df(df,fn)
    return df

def _hop_overlap_summary(navitype, ontology, year, k, params):

    if params['rel'] == 'O':

        fn_hop_overlap = os.path.join(params['cs'], 'hopsoverlap', navitype, '{}_{}_{}HOP.mtx'.format(ontology, year, k))
        m = load_sparse_matrix(fn_hop_overlap)
        fn_hop = os.path.join(params['on'], 'hops', '{}_{}_{}HOP.mtx'.format(ontology.upper(), year, k))
        Ao = load_sparse_matrix(fn_hop)
        overlap = m.sum() / Ao.sum()

    elif params['rel'] == 'T':

        fn_hop_overlap = os.path.join(params['cs'], 'hopsoverlap', navitype, '{}_{}_{}HOP.mtx'.format(ontology, year, k))
        m = load_sparse_matrix(fn_hop_overlap)
        fn_transitions = os.path.join(params['cs'], 'graph', navitype, '{}_{}.adjlist'.format(ontology.upper(), year))
        Gt = nx.read_weighted_edgelist(fn_transitions)
        overlap = m.sum() / Gt.number_of_edges()

    elif params['rel'] == 'MT':

        fn_hop_overlap_weighted = os.path.join(params['cs'], 'hopsoverlap', navitype, '{}_{}_{}HOP_weighted.mtx'.format(ontology.upper(), year, k))
        m = load_sparse_matrix(fn_hop_overlap_weighted)
        fn_transitions = os.path.join(params['cs'], 'graph', navitype, '{}_{}.adjlist'.format(ontology.upper(), year))
        Gt = nx.read_weighted_edgelist(fn_transitions)
        overlap = m.sum() / Gt.size(weight='weight')

    elif params['rel'] == 'RAW':
        overlap = m.sum()

    return (ontology,k,navitype,overlap)

###########################################################################
# MAIN
###########################################################################

if __name__ == '__main__':
    params = utils.get_params(MUST)
    LOGFILE = LOGFILE.replace('<date>', datetime.datetime.now().strftime("%Y-%m-%d")).replace('<opt>',params['opt'])
    logheader(params)

    if params['opt'] == 'init':
        utils.validate_params(params, ['cslog', 'cs'])
        init_analytics(params)

    elif params['opt'] == 'filtering':
        utils.validate_params(params,['cslog','cs'])
        df = filtering(params)

    elif params['opt'] == 'validation':
        utils.validate_params(params,['cslog','on','cs'])
        df = validation(params)

    elif params['opt'] == 'concepts':
        utils.validate_params(params,['cs'])
        plot_concepts_distribution(params)

    elif params['opt'] == 'degreedist':
        utils.validate_params(params, ['cs', 'on'])
        plot_degree_distributions(params)

    elif params['opt'] == 'metadata':
        utils.validate_params(params, ['cs', 'on'])
        valid_metadata(params)

    elif params['opt'] == 'toprequests':
        utils.validate_params(params, ['cs', 'on', 'topk'])
        top_requests(params)

    elif params['opt'] == 'plotnavitypes':
        utils.validate_params(params, ['cs', 'on'])
        plot_session_navitypes(params)

    elif params['opt'] == 'original':
        df = load_clickstreams(params)
        fn = os.path.join(params['output'], 'original_clickstreams.csv')
        df.to_csv(fn)

    elif params['opt'] == 'sessions':
        utils.validate_params(params, ['cslog','cs','year','set','on'])
        generate_session_ids(params)

    elif params['opt'] == 'plotsessions':
        utils.validate_params(params, ['cslog','cs','year','set'])
        plot_sessions(params)

    elif params['opt'] == 'transitions':
        utils.validate_params(params, ['cslog','cs','year','set','topk','withinonto', 'navitype'])
        transitions(params)

    elif params['opt'] == 'plottransitions':
        utils.validate_params(params, ['cslog','cs','year','set','topk','withinonto', 'navitype'])
        plot_transitions(params)

    elif params['opt'] == 'summary':
        utils.validate_params(params, ['cslog', 'on', 'cs', 'year', 'set', 'topk', 'withinonto', 'navitype'])
        summary(params)

    elif params['opt'] == 'hops':
        utils.validate_params(params, ['on', 'year','cs', 'set','topk','withinonto', 'navitype'])
        create_hops_adj(params)

    elif params['opt'] == 'hopsoverlap':
        utils.validate_params(params, ['on', 'year', 'cs', 'navitype'])
        hops_overlap(params)

    elif params['opt'] == 'hopsoverlapsummary':
        utils.validate_params(params, ['on', 'year', 'cs','navitype','rel'])
        hops_overlap_summary(params)


