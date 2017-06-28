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

from os.path import join
import os
import json
import subprocess
import shlex
import re
import logging

# This file contains all the utility functions required for performance test module


def get_families(current_algo, ml_algo):
    """
    Return: List of families given input algorithm

    """

    family_list = []
    for family, algos in ml_algo.items():
        if current_algo in algos:
            family_list.append(family)
    return family_list


def split_rowcol(matrix_dim):
    """
    Return: matrix row, column on input string (e.g. 10k_1k)

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

    with open(write_path, 'w') as input_file:
        json.dump(config_dict, input_file, indent=4)


def config_reader(read_path):
    """
    Return: configuration dictionary on reading the json file

    """

    with open(read_path, 'r') as input_file:
        conf_file = json.load(input_file)

    return conf_file


def create_dir(directory):
    """
    Create directory given path if the directory does not exist already

    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_existence(path):
    """
    Return: Boolean check if the file _SUCCESS exists

    """

    full_path = join(path, '_SUCCESS')
    exist = os.path.isfile(full_path)
    return exist


def exec_dml_and_parse_time(exec_type, file_name, args, time=True):
    """
    This function is responsible of execution of input arguments
    via python sub process. We also extract time obtained from the output of this subprocess.

    exec_type: String

    file_name: String

    args: Dictionary

    time: Boolean (default=True)

    """

    algorithm = file_name + '.dml'
    if exec_type == 'singlenode':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-standalone.py')

        args = ''.join(['{} {}'.format(k, v) for k, v in args.items()])
        cmd = [exec_script, algorithm, args]
        cmd_string = ' '.join(cmd)

    if exec_type == 'hybrid_spark':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-spark-submit.py')
        args = ''.join(['{} {}'.format(k, v) for k, v in args.items()])
        cmd = [exec_script, '-f', algorithm, args]
        cmd_string = ' '.join(cmd)

    # Debug
    # print(cmd_string)

    # Subprocess to execute input arguments
    # proc1_log contains the shell output which is used for time parsing
    proc1 = subprocess.Popen(shlex.split(cmd_string), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    if time:
        proc1_log = []
        while proc1.poll() is None:
            raw_std_out = proc1.stdout.readline()
            decode_raw = raw_std_out.decode('ascii').strip()
            proc1_log.append(decode_raw)
            logging.log(10, decode_raw)

        out1, err1 = proc1.communicate()

        if "Error" in str(err1):
            print('Error Found in {}'.format(file_name))
            total_time = 'failure'
        else:
            total_time = parse_time(proc1_log)
    else:
        total_time = 'not_specified'

    return total_time


def parse_time(raw_logs):
    """
    Return: Time based on rawlogs received

    """
    # Debug
    # print(raw_logs)

    for line in raw_logs:
        if line.startswith('Total execution time'):
            extract_time = re.findall(r'\d+', line)
            total_time = '.'.join(extract_time)
            return total_time

    return 'time_not_found'


def get_config(file_path):
    """
    The purpose of this function is to extract useful information from the folder name.

    file_path : String
    Input file path

    return: matrix type and matrix dim based

    """

    folder_name = file_path.split('/')[-1]
    algo_prop = folder_name.split('.')
    mat_type = algo_prop[1]
    mat_dim = algo_prop[2]

    try:
        intercept = algo_prop[3]
    except IndexError:
        intercept = 'none'

    return mat_type, mat_dim, intercept


def exec_test_data(exec_type, path):
    """
    Creates the test data split from the given input path

    exec_type : String
    This string contains the execution type singlenode/ hybrid_spark

    path : String
    Location of the input folder to pick X and Y

    """
    systemml_home = os.environ.get('SYSTEMML_HOME')
    test_split_script = join(systemml_home, 'scripts', 'perftest', 'extractTestData')
    X = join(path, 'X.data')
    Y = join(path, 'Y.data')
    X_test = join(path, 'X_test.data')
    Y_test = join(path, 'Y_test.data')

    args = {'-args': ' '.join([X, Y, X_test, Y_test, 'csv'])}
    exec_dml_and_parse_time(exec_type, test_split_script, args, False)


def check_predict(current_algo, ML_PREDICT):
    if current_algo in ML_PREDICT.keys():
        return True
    else:
        return False
