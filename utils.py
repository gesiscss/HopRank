__author__ = 'lisette.espin'

import getopt
import sys
from collections import defaultdict
import networkx as nx

# def get_params_ontologies():
#     str = 'ontologies.py -p <ontologies_path> -o <outputfile>'
#     try:
#         opts, args = getopt.getopt(sys.argv,"hp:o:",["ipath=","ofile="])
#     except getopt.GetoptError:
#         print(str)
#         sys.exit(2)
#
#     params = {}
#     for opt, arg in opts:
#         if opt == '-h':
#             print(str)
#             sys.exit()
#         else:
#             params[opt] = arg
#
#     return params

def get_params(must_args):
    if len(sys.argv) == 1:
        print('No arguments: python <script>.py -argid valueid')
        sys.exit(2)

    key = None
    params = defaultdict(None)
    for arg in sys.argv:

        if str(arg).startswith('-'):
            key = arg.replace('--','').replace('-','')
            continue

        if key is not None:
            params[key] = arg
            key = None

    missing = []
    for ma in must_args:
        if ma not in params:
            missing.append(ma)
    if len(missing) > 0:
        print('These params are missing: \n{}'.format('\n'.join(missing)))
        sys.exit(2)
    return params

def validate_params(params,musts):
    i = set(params.keys()).intersection(set(musts))
    if len(i) != len(musts):
        print('params missing: {}'.format([m for m in musts if m not in i]))
        sys.exit(2)
