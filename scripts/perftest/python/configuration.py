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

import os
from os.path import join
from utils import split_rowcol, config_writer, create_dir
import sys
import logging

# This directory contains all the configuration parameters required for all algorithms
# defined in our performance test suit, our test suite contains three types of configurations
# datagen, train and predict configurations

# Kmeans,30K_1K

format = 'csv'
data_gen = {'Kmeans': 'clustering_dg'}


def clustering_dg(datagen_dir, matrix_config):
    row, col = split_rowcol(matrix_config)

    config = dict(nr=row, nf=col, nc='5', dc='10.0', dr='1.0',
                  fbf='100.0', cbf='100.0', X='X.data', C='C.data', Y='Y.data',
                  YbyC='YbyC.data', fmt=format)

    file_name = '_'.join(['clustering', str(matrix_config) + '.json'])
    write_path = join(datagen_dir, file_name)
    config_writer(write_path, config)

    return write_path


def init_conf(algorithm, datagen_dir, matrix_config):

    algo_dg = data_gen[algorithm]
    conf_path = globals()[algo_dg](datagen_dir, matrix_config)
    return conf_path


def config_bundle(algorithm, matrix_type, matrix_shape):
    config_packet = {}

    for algo in algorithm:
        if algo == 'Kmeans':
            config_packet[algo] = matrix_shape
        else:
            config = list(itertools.product(matrix_shape, matrix_type))
            config_packet[algo] = config

    return config_packet


def gen_data_config(algorithm, matrix_type, matrix_shape, temp_dir):

    conf_files = {}

    # Create data-gen dir
    data_gen_dir = join(temp_dir, 'data-gen')
    create_dir(data_gen_dir)

    conf_dict = config_bundle(algorithm, matrix_type, matrix_shape)

    for algo, matrix_config in conf_dict.items():
        conf_files[algo] = []
        for mat_conf in matrix_config:
            conf_path = init_conf(algo, data_gen_dir, mat_conf)
            conf_files[algo].append(conf_path)
            #conf_files.append(conf_path)

    # Return the list of config files to execute
    return conf_files



