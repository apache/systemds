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
{% end comment %}
-->

# Python Script Generator for Builtin Functions - Design Document

[TOC]

Discussed in PR [https://github.com/apache/systemds/pull/1135]

## Design

This document describes the initial design of the `python-generator` for builtin functions.

This generator allows us to automatically generate both the API Front End and the documentation from the DML builtin functions. The files of the builtin functions which we want to use are stored in the folder `scripts\builtin`.

## Goals

The goal of this project is to use the dml files in `scripts\builtin` to automatically build python scripts for the systemDS python API. Based on the dml file we should be able create a documentation file for the builtin function as well as generate the python script. We would use the dml file to retrieve the parameters needed, their default values, if specified, as well as the function definition.

## Limitations

At first we have to define a subset of builtin function which we want to generate from our generator script, since we cannot ensure that all dml scripts within the folder `scripts\builtin` follow our assumed file structure.

## DML File Structure

For the DML script structure we assume that each file has to own a header section. We assume the header to start with a line that indicates the type of the header, this line has to contain exactly one of the following substrings  [`INPUT`, `RETURN`, `OUTPUT`] followed by a newline character. Where for example `INPUT` would indicate a header for the input parameters of the function. Subsequently to that line we assume a divider line, which contains an arbitrary number of `-` ending with a newline character. After this divider line we assume to find a header line for the parameters. We assume that this line contains all of these substrings [`NAME`, `TYPE`, `DEFAULT`, `MEANING`], where each substring is separated from each other by an arbitrary number of space characters and the line has to end with a newline character. Since we always assume this exact order the content of this header line is irrelevant to us and we therefore won’t check it. Following this line we again assume to find a divider line. Subsequently we now assume to find all the information for the parameters of the type which we defined in the first line (eg. `INPUT`). For each line we assume that the first word is the `NAME` of the variable, the second word we will find is the `TYPE` of the variable, the third word indicates the `DEFAULT` value of the variable, where an arbitrary number of `-` indicates that the parameter is required for the function. Everything after the third word is the description. We assume that the `NAME`, `TYPE` and `DEFAULT` do not contain any separation characters such as the space characters and that each word is separated by a space character. Each line is expected to end with a newline character. We assume the end of a parameter list to be indicated by a divider line. If for the following lines do not contain any substring of [`INPUT`, `RETURN`, `OUTPUT`]  we assume that we reached the end of the header section.

We will use the function name from the filename. For the function definition of the builtin function we assume that it starts with `m_`, the parameters, return values and their respective types will be parsed from the function definition. Optional parameters for the `**kwargs` dict are recognized by having a default value.

Bellow we provided a example for a valid header and function definition:

```R
# INPUT PARAMETERS:
# ----------------------------------------------------------------------------
# NAME                              TYPE      DEFAULT  MEANING
# ----------------------------------------------------------------------------
# X                                 Double    ---      The input Matrix to do KMeans on.
# k                                 Int       ---      Number of centroids
# runs                              Int       10       Number of runs (with different initial centroids)
# max_iter                          Int       1000     Maximum number of iterations per run
# eps                               Double    0.000001 Tolerance (epsilon) for WCSS change ratio
# is_verbose                        Boolean   FALSE    do not print per-iteration stats
# avg_sample_size_per_centroid      Int       50       Average number of records per centroid in data samples
# seed                              Int       -1       The seed used for initial sampling. If set to -1 random seeds are selected. 
# ----------------------------------------------------------------------------
#
# RETURN VALUES
# ----------------------------------------------------------------------------
# NAME     TYPE      DEFAULT  MEANING
# ----------------------------------------------------------------------------
# Y        String    "Y.mtx"  The mapping of records to centroids
# C        String    "C.mtx"  The output matrix with the centroids
# ----------------------------------------------------------------------------


m_kmeans = function(Matrix[Double] X, Integer k = 10, Integer runs = 10, Integer max_iter = 1000,
                    Double eps = 0.000001, Boolean is_verbose = FALSE, Integer avg_sample_size_per_centroid = 50,
                    Integer seed = -1)
           return (Matrix[Double] C, Matrix[Double] Y)
```

### Grammar

The following grammar defines the header section for a DML script. Uppercase words are tokens and are defined as a sequence of other tokens and terminals. A | symbol indicates a choice and terminals are written in a code block like this `example`.

| Symbol  | Definition                                                   |
| ------- | ------------------------------------------------------------ |
| DML     | HEAD `\n` \| HEAD HEAD                                       |
| HEAD    | `INPUT` `\n` DIVIDER PHEADER DIVIDER PARAM | `RETURN` `\n` DIVIDER PHEADER DIVIDER PARAM | `OUTPUT` `\n` DIVIDER PHEADER DIVIDER PARAM |
| PHEADER | `NAME` SPACE `TYPE` SPACE `DEFAULT` SPACE `MEANING` `\n`     |
| DIVIDER | `-` `\n` | `-` DIVIDER                                       |
| PARAM   | TEXT SPACE TEXT SPACE TEXT SPACE TEXTS DIVIDER               |
| TEXT    | { A string of printable ASCII characters, without newlines or spaces} |
| TEXTS   | TEXT SPACE TEXTS\| TEXT `\n` PARAM\| TEXT `\n`               |
| SPACE   | { Sequence of one or more space characters }                 |

## Python Script Structure

An `OperationNode` represents an operation that executes in SystemDS. The goal of the python script is to create a valid `OperationNode` with the correct parameters and the respective function name of the function we want to call in SystemDS. In this section we first want to define the required parameters of the `OperationNode`.

The following description was taken from the python [API documentation](https://apache.github.io/systemds/api/python/api/operator/operation_node.html).
A `OperationNode` requires:

- **sds_context** – The SystemDS context for performing the operations
- **operation** – The name of the DML function to execute
- **unnamed_input_nodes** – inputs identified by their position, not name
- **named_input_nodes** – inputs with their respective parameter name
- **output_type** – type of the output in DML (double, matrix etc.)
- **is_python_local_data** – if the data is local in python e.g. Numpy arrays
- **number_of_outputs** – If set to other value than 1 then it is expected that this operation node returns multiple values. If set remember to set the output_types value as well.
- **output_types** – The types of output in a multi output scenario. Default is None, and means every multi output is a matrix.

Lets take the DML script of `kmeans` and its function definition as an example.

```R
m_kmeans = function(Matrix[Double] X, Integer k = 10, Integer runs = 10, Integer max_iter = 1000,
                    Double eps = 0.000001, Boolean is_verbose = FALSE, Integer avg_sample_size_per_centroid = 50,
                    Integer seed = -1)
           return (Matrix[Double] C, Matrix[Double] Y)
```

Using the dml function definition we can create the `OperationNode` as follows:

`Matrix[Double] X` indicates a required parameter with no default value. Since `X` is a `Matrix` we will specify the python input parameter type for `x` to be an `OperationNode`. Since all the other parameters are optional input parameters we will assign them to `‘**kwargs: Dict[str, VALID_INPUT_TYPES]`. The return type of the python function is `OperationNode` because an `OperationNode` object always returns an `OperationNode` object as defined in the python API documentation. Following this specification we would generate a python function definition that looks as follows:

```python
def kmeans(x: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
```

After the function definition the python script file structure contains a multi line-string comment with parameter information which we will parse from the dml script header.

Subsequently some input parameter checks are performed. `x._check_matrix_op()` should we called on `OperationNode` typed inputs to ensure that it is a `matrix`.

In the next step the `OperationNode` requires a dictionary of the input parameters. For the required input parameters we have to assign a key which we can take from the dml script function definition. All optional parameters can simply be updated to the dictionary.

The code below shows how a python function should look like when parsed from the dml script.

````python
def kmeans(x: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs KMeans on matrix input.

    :param x: Input dataset to perform K-Means on.
    :param k: The number of centroids to use for the algorithm.
    :param runs: The number of concurrent instances of K-Means to run (with different initial centroids).
    :param max_iter: The maximum number of iterations to run the K-Means algorithm for.
    :param eps: Tolerance for the algorithm to declare convergence using WCSS change ratio.
    :param is_verbose: Boolean flag if the algorithm should be run in a verbose manner.
    :param avg_sample_size_per_centroid: The average number of records per centroid in the data samples.
    :return: `OperationNode` List containing two outputs 1. the clusters, 2 the cluster ID associated with each row in x.
    """

    x._check_matrix_op()
    if x.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x.shape))

    if 'k' in kwargs.keys() and kwargs.get('k') < 1:
        raise ValueError(
            "Invalid number of clusters in K-Means, number must be integer above 0")

    params_dict = {'X': x}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'kmeans', named_input_nodes=params_dict, output_type=OutputType.LIST,
        number_of_outputs=2)
````

## Steps

1. Implement parser for dml function definitions (or use existing parser)
2. Implement parser for dml function headers
3. Implement generator, that generates Python API functions from parsed dml function definitions
4. Implement generator, that generates multi-line string comment for Python API functions from parsed function headers
5. Implement generator, that generates python files with license header and can use an arbitrary amount of string parameters, in our case for python code and comments
6. Include subprocess call for the builtin-api-generator in `src/main/python/create_python_dist.py`

## Generator Folder Structure

We plan on creating a folder called `generator` in `src/main/python/systemds`. In this folder we will place our `generator.py`, a `parser.py` and a folder called `scripts/builtin` where we will place the generated python scripts.

In summary we will create the following folder structures:

1. `src/main/python/systemds/generator/generator.py`
2. `src/main/python/systemds/generator/parser.py`
3. `src/main/python/systemds/generator/scripts/builtin/`
