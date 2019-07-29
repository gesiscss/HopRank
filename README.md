
#### [ 2019 ] ####
HopRank: How Semantic Structure Influences Teleportation in PageRank (A Case Study on BioPortal)
------
Lisette Espín-Noboa, Florian Lemmerich, Simon Walk, Markus Strohmaier, and Mark Musen. 2019. 
HopRank: How Semantic Structure Influences Teleportation in PageRank (A Case Study on BioPortal). 
In Proceedings of the 2019 World Wide Web Conference (WWW '19), May 13–17, 2019, San Francisco, CA, USA. 
ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3308558.3313487


### Jupyter Notebooks ###
- **`1. BioPortal (pre-processing).ipynb`**:
Executing necessary scripts to clean and pre-process: Load clickstream, infer types of navigation using HTTP headers, lag distribution, defining sessions, valid requests: (HTTP-200, valid navitype, non-empty ontology and concept).

- **`2. BioPortal (valid).ipynb`**:
Executing necessary scripts to clean and pre-process: cross-validation of ontologies and concepts in clickstream (they should exist). Defining transitions.

- **`3. BioPortal (khop-matrices-lcc).ipynb`**:
Executing necessary scripts to create k-hop matrices for every ontology.

- **`4. BioPortal (distance-matrices-lcc).ipynb`**:
Executing necessary scripts to create distance matrices for every ontology (aggregation of k-hop matrices, memory expensive).

- **`5. HopRank (empirical-lcc).ipynb`**:
Executing necessary scripts to detect overlaps between transitions and ontology structure. Calculation of HopPortation vectors for every ontology and navigation type (betas)

- **`6. HopRank (model fitting-lcc).ipynb`**:
Executing necessary scripts to fit models: HopRank, Markov chain, Random Walk (with different damping factors: 0.0, 1.0, 0.85, empirical), Preferential Attachment (degree), Gravitational (degree/distance^2). Plot BIC scores, and best model (lowest BIC). Includes toy-example.



### Main Libraries ###

- **`org/gesis/libs/bioportal/clickstream.py`**:
Functions to pre-process clickstreams. Assigns sessions, navigation types to every request.

- **`org/gesis/libs/bioportal/ontology.py`**: 
Ontology class. Methods to load raw csv files, converting DataFrame to DiGraph (networkx) and adjacency matrix (csr_matrix).

- **`org/gesis/libs/bioportal/submission.py`**:
Functions to pre-process ontology submissions and retrieve last ontology per year.

- **`org/gesis/libs/bioportal/transition.py`**:
Transition class. Methods to load clickstreams per ontology, converting DataFrame to DiGraph (networkx) and adjacency matrix 
(csr_matrix).

- **`org/gesis/libs/models/*.py`**:
Class for 6 different models of navigation: HopRank, Markov chain, Random Walk, Preferential Attachment (degree), Gravitational (degree/distance^2). Computes loglikelihood, BIC, AIC.


[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/gesiscss/HopRank/master)
