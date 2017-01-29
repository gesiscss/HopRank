__author__ = 'espin'

import utils
from utils import Utils

URI_CATEGORIES = 'http://data.bioontology.org/ontologies/:ontology/categories'
URI_GOOGLE_ANALYTICS = 'http://data.bioontology.org/ontologies/:ontology/analytics'
URI_MAPPINGS = 'http://data.bioontology.org/ontologies/:ontology/mappings'
URI_TOTAL_MAPPINGS = 'http://http://data.bioontology.org/mappings/statistics/ontologies/:ontology'
URI_ONTOLOGY_CLASSES = 'http://data.bioontology.org/ontologies/:ontology/classes'
URI_ONTOLOGY_SUBMISSIONS = 'http://data.bioontology.org/ontologies/:ontology/submissions'
URI_ONTOLOGY_PROJECTS = 'http://data.bioontology.org/ontologies/:ontology/projects'
URI_ONTOLOGY_METRICS = 'http://data.bioontology.org/ontologies/:ontology/metrics'
URI_ONTOLOGY_SUMISSION_METRIC = 'http://data.bioontology.org/ontologies/:ontology/submissions/:submissionid/metrics'
URI_ONTOLOGIES = 'http://data.bioontology.org/ontologies'

API_KEY = '4bb8b0bd-4160-41ba-add9-eb548ab3c6df'

class Ontology(Utils):

    def __init__(self):
        print ''


if __name__ == '__main__':
    ontology = utils.getParameter(1)
    opt = utils.getParameter(2)

#
# [2017-01-22 00:29:45.236075] apirequests-ontology:MESH-within_ontologies-directlink
# [2017-01-22 00:29:48.219422] === TRANSITIONS ===
# [2017-01-22 00:29:48.220925] TRANS MATCHING: 798 out of 1467707.0 (0.0543705249072%)
# [2017-01-22 00:29:48.222166] TRANS DIS-MATCHING: 1466909.0 out of 1467707.0 (99.9456294751%)
# [2017-01-22 00:29:48.222971] === ONTOLOGY ===
# [2017-01-22 00:29:48.223718] ONTO MATCHING: 798 out of 268263.0 (0.297469274555%)
# [2017-01-22 00:29:48.224393] ONTO DIS-MATCHING: 267465.0 out of 268263.0 (99.7025307254%)
# [2017-01-22 00:29:49.140648] END: 22.6706869602 secods.


# [2017-01-22 00:45:56.439717] apirequests-ontology:MESH-within_ontologies-directlink
# [2017-01-22 00:46:00.936054] === TRANSITIONS ===
# [2017-01-22 00:46:00.937439] TRANS MATCHING: 309 out of 1467707.0 (0.0210532483663%)
# [2017-01-22 00:46:00.938655] TRANS DIS-MATCHING: 1467398.0 out of 1467707.0 (99.9789467516%)
# [2017-01-22 00:46:00.939711] === ONTOLOGY ===
# [2017-01-22 00:46:00.941105] ONTO MATCHING: 309 out of 37909.0 (0.815109868369%)
# [2017-01-22 00:46:00.942206] ONTO DIS-MATCHING: 37600.0 out of 37909.0 (99.1848901316%)
# [2017-01-22 00:46:01.758684] END: 130.328361988 secods.


# [2017-01-22 01:06:40.182741] apirequests-ontology:MESH-within_ontologies-directlink
# [2017-01-22 01:06:42.533014] === TRANSITIONS ===
# [2017-01-22 01:06:42.533953] TRANS MATCHING: 798 out of 1467707.0 (0.0543705249072%)
# [2017-01-22 01:06:42.534888] TRANS DIS-MATCHING: 1466909.0 out of 1467707.0 (99.9456294751%)
# [2017-01-22 01:06:42.535453] === ONTOLOGY ===
# [2017-01-22 01:06:42.536018] ONTO MATCHING: 798 out of 268263.0 (0.297469274555%)
# [2017-01-22 01:06:42.536551] ONTO DIS-MATCHING: 267465.0 out of 268263.0 (99.7025307254%)
# [2017-01-22 01:06:43.252490] END: 149.631386995 secods.
