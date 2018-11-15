
### Scripts ###

- **`org/gesis/libs/bioportal/clickstreams.py`**:
Functions to pre-process ontology submissions and retrieve last ontology per year.

- **`org/gesis/libs/bioportal/ontologies.py`**: 
Ontology class. Methods to load raw csv files, converting DataFrame to DiGraph (networkx) and adjacency matrix (csr_matrix).

- **`org/gesis/libs/bioportal/clickstreams.py`**:
Functions to pre-process clickstreams. Assigns sessions, navigation types to every request.

- **`org/gesis/libs/bioportal/transitions.py`**:
Transition class. Methods to load clickstreams per ontology, converting DataFrame to DiGraph (networkx) and adjacency matrix 
(csr_matrix).


### Jupyter Notebook ###
- **`BioPortal.ipynb`**:
Executing necessary scripts to clean and pre-process ontologies and clickstreams from BioPortal.

