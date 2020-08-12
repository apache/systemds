---
layout: site
title: Entity Resolution
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

## Pipeline design and primitives

We provide two example scripts, `entity-clustering.dml` and `binary-entity-resolution.dml`. These handle reading input 
files and writing output files and call functions provided in `primitives/pipeline.dml`.

The pipeline design is loosely based on the following paper, but does not use advanced features like multi-probe LSH,
combining embeddings via LSTM or classification via machine learning.

```
Ebraheem, Muhammad, et al. "Distributed representations of tuples for entity resolution."
Proceedings of the VLDB Endowment 11.11 (2018): 1454-1467.
```

### Input files

The provided scripts can read two types of input files. The token file is mandatory since it contains the row identifiers, 
but the embedding file is optional. The actual use of tokens and/or embeddings can be configured via command line parameters 
to the scripts.

##### Token files

This file type is a CSV file with 3 columns. The first column is the string or integer row identifier, the second is the 
string token, and the third is the number of occurences. This simple format is used as a bag-of-words representation.

##### Embedding files

This file type is a CSV matrix file with each row containing arbitrary-dimensional embeddings. The order of row identifiers
is assumed to be the same as in the token file. This saves some computation and storage time, but could be changed with 
some modifications to the example scripts.

### Primitives

While the example scripts may be sufficient for many simple use cases, we aim to provide a toolkit of composable functions
to facilitate more complex tasks. The top-level pipelines are defined as a couple of functions in `primitives/pipeline.dml`.
The goal is that it should be relatively easy to copy one of these pipelines and swap out the primitive functions used
to create a custom pipeline.

To convert the input token file into a bag-of-words contingency table representation, we provide the functions
`convert_frame_tokens_to_matrix_bow` and `convert_frame_tokens_to_matrix_bow_2` in  `primitives/preprocessing.dml`.
The latter is used to compute a compatible contigency table with matching vocabulary for binary entity resolution. 

We provide naive, constant-size blocking and locality-sensitive hashing (LSH) as functions in `primitives/blocking.dml`.

For entity clustering, we only provide a simple clustering approach which makes all connected components in an adjacency
matrix fully connected. This function is located in `primitives/clustering.dml`.

To restore an adjacency matrix to a list of pairs, we provide the functions `untable` and `untable_offset` in 
`primitives/postprocessing.dml`.

Finally, `primitives/evaluation.dml` defines some metrics that can be used to evaluate the performance of the entity
resolution pipelines. They are used in the script `eval-entity-resolution.dml`. 

## Testing and Examples

There is a test data repository that was used to develop these scripts at 
[repo](https://github.com/skogler/systemds-amls-project-data). In the examples below, it is assumed that this repo is 
cloned as `data` in the SystemDS root folder. The data in that repository is sourced from the Uni Leipzig entity resolution 
[benchmark](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution).

### Preprocessing

Since there is no tokenization functionality in SystemDS yet, we provide a Python preprocessing script in the data repository
that tokenizes the text columns and performs some simple embedding lookup using Glove embeddings.

The tokens are written as CSV files to enable Bag-of-Words representations as well as matrices with combined embeddings. D
epending on the type of data, one or the other or a combination of both may be better. The SystemDS DML scripts can be 
called with different parameters to experiment with this.

### Entity Clustering

In this case we detect duplicates within one database. As an example, we use the benchmark dataset Affiliations from Uni Leipzig.
For this dataset, embeddings do not work well since the data is mostly just names. Therefore, we encode it as Bag-of-Words vectors
in the example below. This dataset would benefit from more preprocessing, as simply matching words for all the different kinds of
abbreviations does not work particularly well.

Example command to run on Affiliations dataset:
```
./bin/systemds ./scripts/algorithms/entity-resolution/entity-clustering.dml -nvargs FX=data/affiliationstrings/affiliationstrings_tokens.csv OUT=data/affiliationstrings/affiliationstrings_res.csv store_mapping=FALSE MX=data/affiliationstrings/affiliationstrings_MX.csv use_embeddings=FALSE XE=data/affiliationstrings/affiliationstrings_embeddings.csv
```
Evaluation:
```
./bin/systemds ./scripts/algorithms/entity-resolution/eval-entity-resolution.dml -nvargs FX=data/affiliationstrings/affiliationstrings_res.csv FY=data/affiliationstrings/affiliationstrings_mapping_fixed.csv
```

### Binary Entity Resolution

In this case we detect duplicate pairs of rows between two databases. As an example, we use the benchmark dataset DBLP-ACM from Uni Leipzig.
Embeddings work really well for this dataset, so the results are quite good with an F1 score of 0.89.

Example command to run on DBLP-ACM dataset with embeddings:
```
./bin/systemds ./scripts/algorithms/entity-resolution/binary-entity-resolution.dml -nvargs FY=data/DBLP-ACM/ACM_tokens.csv FX=data/DBLP-ACM/DBLP2_tokens.csv MX=data/DBLP-ACM_MX.csv OUT=data/DBLP-ACM/DBLP-ACM_res.csv XE=data/DBLP-ACM/DBLP2_embeddings.csv YE=data/DBLP-ACM/ACM_embeddings.csv use_embeddings=TRUE
```
Evaluation:
```
./bin/systemds ./scripts/algorithms/entity-resolution/eval-entity-resolution.dml -nvargs FX=data/DBLP-ACM/DBLP-ACM_res.csv FY=data/DBLP-ACM/DBLP-ACM_perfectMapping.csv
```

## Future Work

1. Better clustering algorithms.
    1. Correlation clustering.
    2. Markov clustering.
    3. See [this link](https://dbs.uni-leipzig.de/en/publication/title/comparative_evaluation_of_distributed_clustering_schemes_for_multi_source_entity_resolution) for more approaches.
2. Multi-Probe LSH to improve runtime performance. 
    1. Probably as a SystemDS built-in to be more efficient.
3. Classifier-based matching.
    1. Using an SVM classifier to decide if two tuple are duplicates instead of a threshold for similarity.
4. Better/built-in tokenization.
    1. Implement text tokenization as component of SystemDS.
    2. Offer choice of different preprocessing and tokenization algorithms (e.g. stemming, word-piece tokenization).
5. Better/built-in embeddings.
   1. Implement embedding generation as component of SystemDS.
   2. Use LSTM to compose embeddings.