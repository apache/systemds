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

from functools import reduce
import os
import json


def get_algo(family, ml_algo):
    '''
    This function return all algorithms defined in the current run given family.

    :param family: List of algorithms specified for a run
    :param ml_algo: Dictionary with family and all algorithms within it
    :return: List of algorithms
    '''

    algo = []
    for fam in family:
        algo.append(ml_algo[fam])
    algo_flat = reduce(lambda x, y: x + y, algo)
    return algo_flat


def split_rowcol(matrix_dim):
    '''
    This function returns input matrix dimensions in integer format.

    :param matrix_dim: String that contains matrix dim (e.g 10k_1k)
    :return: List of tuple (e.g [(10000, 1000)])
    '''

    mat_shapes = []
    for dims in matrix_dim:
        k = str(0) * 3
        M = str(0) * 6
        replace_M = dims.replace('M', str(M))
        replace_k = replace_M.replace('k', str(k))
        row, col = replace_k.split('_')
        mat_shapes.append((row, col))

    return mat_shapes


def config_writer(path, config_dict, file_name):
    '''
    This function writes the configuration parameters to a file

    :param path: String which specifies output directory to write file
    :param config_dict: Dictionary that contains configuration parameters
    :param file_name: String file name to be written
    :return:
    '''

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/' + file_name, 'w') as json_file:
        json.dump(config_dict, json_file)
