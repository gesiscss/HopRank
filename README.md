# HopSurfer
Random Surfer biased towards kHOP neighborhoods for Modeling Semantic Foraging in BioPortal.

### Abstract ###
HopSurfer is  a  method  for  modeling user navigation in semantic networks. 
More specifically, given a visible semantic network (e.g., ontology), 
we model navigation paths as Markov chains where states represent nodes 
(e.g., concepts) in the network and the probability of moving to node j is 
conditioned not only by the current node i but also by the distance between 
nodes i and j. This is a variation of a random surfer, where teleportation is not 
at random and the probability of visiting a page j is recalculated after every click. 
The idea behind this model comes from Information Foraging, where users try to reduce
the average cost of getting from one information patch to another,and at the same 
time improve the within-patch foraging results. We observe such behavior in 
BioPortal, an online repository of biological and biomedical ontologies. 
In general, users navigate within the vicinity of a concept page. But they also 
“jump” to semantically distant concepts less frequently. We fit the ontology 
structure into our model, learn the model parameters from the transition matrix 
of clickstreams, and  accurately reproduce the number of transitions between 
different k-HOP neighborhoods. We demonstrate that HopSurfer outperforms the 
baseline models random surfer, and Markov chains in semantic navigation.

