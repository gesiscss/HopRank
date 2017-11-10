# BioPortalStanford
Validating relationships between/within Ontologies in BioPortal


######## NEW CPT 2016 CLICKSTREAM
python ontologies.py /bigdata/lespin/datasets/bioportal/ontologies CPT 2016 /bigdata/lespin/bioportal-ontologies ontograph
python transitionParser.py clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'CPT'}" "{'request_action':'Browse Ontology Class Tree'}" /bigdata/lespin/bioportal-ontologies/
python transitions.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'CPT'}" "{'request_action':'Browse Ontology Class Tree'}"
python prediction.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'CPT'}" "{'request_action':'Browse Ontology Class Tree'}" predictions
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'CPT'}" "{'request_action':'Browse Ontology Class Tree'}" entryexit
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'CPT'}" "{'request_action':'Browse Ontology Class Tree'}" entryexitclass
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'CPT'}" "{'request_action':'Browse Ontology Class Tree'}" linkclass
######## NEW CPT 2016


######## NEW RXNORM 2015 CLICKSTREAM
python ontologies.py /bigdata/lespin/datasets/bioportal/ontologies RXNORM 2016 /bigdata/lespin/bioportal-ontologies ontograph
python transitionParser.py clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'RXNORM'}" "{'request_action':'Browse Ontology Class Tree'}" /bigdata/lespin/bioportal-ontologies/
python transitions.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'RXNORM'}" "{'request_action':'Browse Ontology Class Tree'}"
python prediction.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'RXNORM'}" "{'request_action':'Browse Ontology Class Tree'}" predictions
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'RXNORM'}" "{'request_action':'Browse Ontology Class Tree'}" entryexit
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'RXNORM'}" "{'request_action':'Browse Ontology Class Tree'}" entryexitclass
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'RXNORM'}" "{'request_action':'Browse Ontology Class Tree'}" linkclass
######## NEW RXNORM 2016


######## NEW SNOMEDCT 2016 CLICKSTREAM
python ontologies.py /bigdata/lespin/datasets/bioportal/ontologies SNOMEDCT 2016 /bigdata/lespin/bioportal-ontologies ontograph
python transitionParser.py clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'SNOMEDCT'}" "{'request_action':'Browse Ontology Class Tree'}" /bigdata/lespin/bioportal-ontologies/
python transitions.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'SNOMEDCT'}" "{'request_action':'Browse Ontology Class Tree'}"
python prediction.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'SNOMEDCT'}" "{'request_action':'Browse Ontology Class Tree'}" predictions
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'SNOMEDCT'}" "{'request_action':'Browse Ontology Class Tree'}" entryexit
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'SNOMEDCT'}" "{'request_action':'Browse Ontology Class Tree'}" entryexitclass
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'SNOMEDCT'}" "{'request_action':'Browse Ontology Class Tree'}" linkclass
######## NEW SNOMEDCT 2016

######## NEW MESH 2016 CLICKSTREAM
python ontologies.py /bigdata/lespin/datasets/bioportal/ontologies MESH 2016 /bigdata/lespin/bioportal-ontologies ontograph
python transitionParser.py clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'MESH'}" "{'request_action':'Browse Ontology Class Tree'}" /bigdata/lespin/bioportal-ontologies/
python transitions.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'MESH'}" "{'request_action':'Browse Ontology Class Tree'}"
python prediction.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'MESH'}" "{'request_action':'Browse Ontology Class Tree'}" predictions
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'MESH'}" "{'request_action':'Browse Ontology Class Tree'}" entryexit
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'MESH'}" "{'request_action':'Browse Ontology Class Tree'}" entryexitclass
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'MESH'}" "{'request_action':'Browse Ontology Class Tree'}" linkclass

######## NEW NCIT 2016 CLICKSTREAM
python ontologies.py /bigdata/lespin/datasets/bioportal/ontologies NCIT 2016 /bigdata/lespin/bioportal-ontologies ontograph
python transitionParser.py clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'NCIT'}" "{'request_action':'Browse Ontology Class Tree'}" /bigdata/lespin/bioportal-ontologies/
python transitions.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'NCIT'}" "{'request_action':'Browse Ontology Class Tree'}"
python prediction.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'NCIT'}" "{'request_action':'Browse Ontology Class Tree'}" predictions
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'NCIT'}" "{'request_action':'Browse Ontology Class Tree'}" entryexit
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'NCIT'}" "{'request_action':'Browse Ontology Class Tree'}" entryexitclass
python regression.py /bigdata/lespin/bioportal-ontologies/ clickstream /bigdata/lespin/datasets/bioportal/clickstream/BP_webpage_requests_2016.csv.bz2 ../results/ concept "{'ontology':'NCIT'}" "{'request_action':'Browse Ontology Class Tree'}" linkclass
