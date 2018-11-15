__author__ = "Lisette Espin-Noboa"
__copyright__ = "Copyright 2018, HopRank"
__credits__ = ["Florian Lemmerich", "Markus Strohmaier", "Simon Walk", "Mark Musen"]
__license__ = "GPL"
__version__ = "1.0.3"
__maintainer__ = "Lisette Espin-Noboa"
__email__ = "Lisette.Espin@gesis.org"
__status__ = "Developing"

########################################################################################
# Local Dependencies
########################################################################################
from org.gesis.libs.utils import printf

########################################################################################
# System Dependencies
########################################################################################
import os
import json 

########################################################################################
# Functions
########################################################################################

def get_submissions(fn):
    if not os.path.exists(fn):
        raise ValueError("submission fn does not exist!")
        return None
    try:
        with open(fn,'r') as f:
            obj = json.load(f)
        printf('{} loaded!'.format(fn))
        printf('- {} ontologies'.format(len(obj.keys())))
        printf('- {} years'.format(len(set([year for o,years in obj.items() for year,data in years.items()]))))
               
    except Exception as ex:
        printf(ex)
        printf('ERROR: {} NOT loaded!'.format(fn))

    obj = {k.upper():v for k,v in obj.items()}
    return obj    