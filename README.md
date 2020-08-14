# ImpactNegativeTriples
Source code of the CIKM'20 paper: The Impact of Negative Triple Generation Strategies and Anomalies on Knowledge Graph Completion (https://doi.org/10.1145/3340531.3412023).

A significant part of the code was borrowed from OpenKE (https://github.com/thunlp/OpenKE/). We have refactored several methods and included several strategies for generating negative triples.

# How to run
Use train.py to train a model and test.py to evaluate the model.

# Data
In addition to training, validation and test, the datasets contain a file that stores the predicates that are related based on their neighborhoods (compatible_relations.txt). They also contain another file that stores the anomalies of the predicates (zero means no anomaly).
