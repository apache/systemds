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
import os
from os.path import join
from utils import split_rowcol, config_writer
import sys
import logging

# This file contains configuration settings for data generation
format = 'csv'


def kmeans_datagen(algo_name, matrix_dim, matrix_type, datagen_dir):

    row, col = split_rowcol(matrix_dim)
    path_name = '-'.join([algo_name, matrix_type, str(matrix_dim)])

    full_path = join(datagen_dir, path_name)
    X = join(full_path, 'X.data')
    Y = join(full_path, 'Y.data')
    YbyC = join(full_path, 'YbyC.data')
    C = join(full_path, 'C.data')

    config = dict(nr=row, nf=col, nc='50', dc='10.0', dr='1.0',
                  fbf='100.0', cbf='100.0', X=X, C=C, Y=Y,
                  YbyC=YbyC, fmt=format)

    config_writer(full_path + '.json', config)
    return full_path


def config_packets_datagen(algos, matrix_type, matrix_shape, datagen_dir):

    config_bundle = {}

    for algo in algos:
        if algo == 'Kmeans':
            config = list(itertools.product(matrix_shape, ['none']))
            config_bundle[algo] = config
        else:
            config = list(itertools.product(matrix_shape, matrix_type))
            config_bundle[algo] = config

    for current_algo, configs in config_bundle.items():
        config_bundle[current_algo] = []
        for conf in configs:
            algo_func = current_algo.lower() + '_datagen'
            conf_path = globals()[algo_func](current_algo, conf[0], conf[1], datagen_dir)
            config_bundle[current_algo].append(conf_path)

    return config_bundle
