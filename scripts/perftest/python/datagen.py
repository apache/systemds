#!/usr/bin/env python3
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import itertools
from os.path import join
from utils import split_rowcol, config_writer

# This file contains configuration settings for data generation
DATA_FORMAT = 'csv'

MATRIX_TYPE_DICT = {'dense': '0.9',
                    'sparse': '0.01'}

FAMILY_NO_MATRIX_TYPE = ['clustering', 'stats1', 'stats2']


def multinomial_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['multinomial', matrix_type, str(matrix_dim)])
    full_path = join(datagen_dir, path_name)

    numSamples = row
    numFeatures = col
    sparsity = MATRIX_TYPE_DICT[matrix_type]
    num_categories = '150'
    intercept = '0'
    X = join(full_path, 'X.data')
    Y = join(full_path, 'Y.data')
    fmt = DATA_FORMAT

    config = [numSamples, numFeatures, sparsity, num_categories, intercept,
              X, Y, fmt, '1']

    config_writer(full_path + '.json', config)

    return full_path


def binomial_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['binomial', matrix_type, str(matrix_dim)])
    full_path = join(datagen_dir, path_name)

    numSamples = row
    numFeatures = col
    maxFeatureValue = '5'
    maxWeight = '5'
    loc_weights = join(full_path, 'weight.data')
    loc_data = join(full_path, 'X.data')
    loc_labels = join(full_path, 'Y.data')
    noise = '1'
    intercept = '0'
    sparsity = MATRIX_TYPE_DICT[matrix_type]
    tranform_labels = '1'
    fmt = DATA_FORMAT

    config = [numSamples, numFeatures, maxFeatureValue, maxWeight, loc_weights, loc_data,
              loc_labels, noise, intercept, sparsity, fmt, tranform_labels]
    config_writer(full_path + '.json', config)

    return full_path


def regression1_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['regression1', matrix_type, str(matrix_dim)])
    full_path = join(datagen_dir, path_name)

    numSamples = row
    numFeatures = col
    maxFeatureValue = '5'
    maxWeight = '5'
    loc_weights = join(full_path, 'weight.data')
    loc_data = join(full_path, 'X.data')
    loc_labels = join(full_path, 'Y.data')
    noise = '1'
    intercept = '0'
    sparsity = MATRIX_TYPE_DICT[matrix_type]
    tranform_labels = '1'
    fmt = DATA_FORMAT

    config = [numSamples, numFeatures, maxFeatureValue, maxWeight, loc_weights, loc_data,
              loc_labels, noise, intercept, sparsity, fmt, tranform_labels]
    config_writer(full_path + '.json', config)

    return full_path


def regression2_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['regression2', matrix_type, str(matrix_dim)])
    full_path = join(datagen_dir, path_name)

    numSamples = row
    numFeatures = col
    maxFeatureValue = '5'
    maxWeight = '5'
    loc_weights = join(full_path, 'weight.data')
    loc_data = join(full_path, 'X.data')
    loc_labels = join(full_path, 'Y.data')
    noise = '1'
    intercept = '0'
    sparsity = MATRIX_TYPE_DICT[matrix_type]
    tranform_labels = '1'
    fmt = DATA_FORMAT

    config = [numSamples, numFeatures, maxFeatureValue, maxWeight, loc_weights, loc_data,
              loc_labels, noise, intercept, sparsity, fmt, tranform_labels]
    config_writer(full_path + '.json', config)

    return full_path


def clustering_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['clustering', matrix_type, str(matrix_dim)])

    full_path = join(datagen_dir, path_name)
    X = join(full_path, 'X.data')
    Y = join(full_path, 'Y.data')
    YbyC = join(full_path, 'YbyC.data')
    C = join(full_path, 'C.data')
    nc = '50'
    dc = '10.0'
    dr = '1.0'
    fbf = '100.0'
    cbf = '100.0'

    config = dict(nr=row, nf=col, nc=nc, dc=dc, dr=dr, fbf=fbf, cbf=cbf, X=X, C=C, Y=Y,
                  YbyC=YbyC, fmt=DATA_FORMAT)

    config_writer(full_path + '.json', config)
    return full_path


def stats1_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['stats1', matrix_type, str(matrix_dim)])
    full_path = join(datagen_dir, path_name)

    DATA = join(full_path, 'X.data')
    TYPES = join(full_path, 'types')
    TYPES1 = join(full_path, 'set1.types')
    TYPES2 = join(full_path, 'set2.types')
    INDEX1 = join(full_path, 'set1.indices')
    INDEX2 = join(full_path, 'set2.indices')
    MAXDOMAIN = '1100'
    SETSIZE = '20'
    LABELSETSIZE = '10'

    # NC should be less than C and more than num0
    # NC = 10 (old value)
    # num0 = NC/2
    # num0 < NC < C
    # NC = C/2
    NC = int(int(col)/2)

    config = dict(R=row, C=col, NC=NC, MAXDOMAIN=MAXDOMAIN, DATA=DATA, TYPES=TYPES, SETSIZE=SETSIZE,
                  LABELSETSIZE=LABELSETSIZE, TYPES1=TYPES1, TYPES2=TYPES2, INDEX1=INDEX1, INDEX2=INDEX2,
                  fmt=DATA_FORMAT)

    config_writer(full_path + '.json', config)

    return full_path


def stats2_datagen(matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '.'.join(['stats2', matrix_type, str(matrix_dim)])
    full_path = join(datagen_dir, path_name)

    D = join(full_path, 'X.data')
    Xcid = join(full_path, 'Xcid.data')
    Ycid = join(full_path, 'Ycid.data')
    A = join(full_path, 'A.data')

    config = dict(nr=row, nf=col, D=D, Xcid=Xcid, Ycid=Ycid,
                  A=A, fmt=DATA_FORMAT)

    config_writer(full_path + '.json', config)
    return full_path


def config_packets_datagen(algo_payload, matrix_type, matrix_shape, datagen_dir):
    """
    This function has two responsibilities. Generate the configuration files for
    datagen algorithms and return a dictionary that will be used for execution.

    algo_payload : List of tuples
    The first tuple index contains algorithm name and the second index contains
    family type.

    matrix_type: String
    Type of matrix to generate e.g dense or sparse

    matrix_shape: String
    Shape of matrix to generate e.g 100k_10

    return: Dictionary {string: list}
    This dictionary contains algorithms to be executed as keys and the path of configuration
    json files to be executed list of values.
    """

    config_bundle = {}

    distinct_families = set(map(lambda x: x[1], algo_payload))

    # Cross Product of all configurations
    for current_family in distinct_families:
        if current_family in FAMILY_NO_MATRIX_TYPE:
            config = list(itertools.product(matrix_shape, ['dense']))
            config_bundle[current_family] = config
        else:
            config = list(itertools.product(matrix_shape, matrix_type))
            # clustering : [[10k_1, dense], [10k_2, dense], ...]
            config_bundle[current_family] = config

    config_packets = {}
    for current_family, configs in config_bundle.items():
        config_packets[current_family] = []
        for size, type in configs:
            family_func = current_family.lower() + '_datagen'
            conf_path = globals()[family_func](size, type, datagen_dir)
            config_packets[current_family].append(conf_path)

    return config_packets
