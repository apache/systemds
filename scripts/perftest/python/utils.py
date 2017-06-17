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
from os.path import join
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT

# This file contains all the utility functions required for performance test module


def get_algo(family, ml_algo):
    """
    Return a list of algorithms given family

    """

    algo = []
    for fam in family:
        algo.append(ml_algo[fam])
    algo_flat = reduce(lambda x, y: x + y, algo)
    return algo_flat


def get_family(algos, ml_algo):
    """
    Return family given algorithm

    """

    for algo in algos:
        for key, value in ml_algo.items():
            if algo in value:
                family = key
    return family


def split_rowcol(matrix_dim):
    """
    Return matrix row, column on input string (e.g. 10k_1k)

    """

    k = str(0) * 3
    M = str(0) * 6
    replace_M = matrix_dim.replace('M', str(M))
    replace_k = replace_M.replace('k', str(k))
    row, col = replace_k.split('_')
    return row, col


def config_writer(write_path, config_dict):
    """
    Writes the dictionary as an configuration json file to the give path

    """

    with open(write_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4)

    return None


def config_reader(read_path):
    """
    Return configuration dictionary on reading the json file

    """

    with open(read_path, 'r') as f:
        conf_file = json.load(f)

    return conf_file


def create_dir(directory):
    """
    Create directory given path if the directory does not exist already

    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def exec_func(exec_type, file_name, args):
    """
    This function is responsible of execution

    :param exec_type: String which can be either spark or singlenode
    :param algorithm: String which has the current algorithm
    :param args: String containing key value arguments
    :return: Array with metrics required for logging
    """

    # TODO
    # If code fails return failure

    algorithm = file_name + '.dml'
    if exec_type == 'singlenode':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-standalone.py')
        cmd = [exec_script, algorithm, '-nvargs', args]
        cmd_string = ' '.join(cmd)
        print(cmd_string)

    if exec_type == 'hybrid_spark':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-spark-submit.py')
        cmd = [exec_script, '-f', algorithm, '-nvargs', args]
        cmd_string = ' '.join(cmd)

    time_start = time.time()
    return_code = subprocess.call(cmd_string, shell=True)
    total_time = time.time() - time_start - 3

    return total_time


def get_config(file_path):
    """
    Returns matrix type and matrix dim based.

    """

    path_split = file_path.split('/')[-1]
    algo_prop = path_split.split('-')
    mat_type = algo_prop[1]
    mat_dim = algo_prop[2].split('.')[0]
    return mat_type, mat_dim