#!/usr/bin/env python3
# -------------------------------------------------------------
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
# -------------------------------------------------------------

from os.path import join
import os
import json
import re
import sys
import getpass
from utils_exec import subprocess_exec
from utils_fs import check_hdfs_path

# This file contains all misc utility functions required by performance test module


def split_config_args(args):
    """
    Based on configuration parameters passed build configuration dictionary used by subprocess

    args: Dictionary
    All parameters passed in

    return: Dictionary, Dictionary, Dictionary
    3 dictionaries - one for perf tests, one for systemml specific args, one for backend options    
    """

    perftest_args_dict = {}

    perftest_args_dict['family'] = args['family']
    perftest_args_dict['algo'] = args['algo']
    perftest_args_dict['exec_type'] = args['exec_type']
    perftest_args_dict['mat_type'] = args['mat_type']
    perftest_args_dict['mat_shape'] = args['mat_shape']
    perftest_args_dict['config_dir'] = args['config_dir']
    perftest_args_dict['filename'] = args['filename']
    perftest_args_dict['mode'] = args['mode']
    perftest_args_dict['temp_dir'] = args['temp_dir']
    perftest_args_dict['file_system_type'] = args['file_system_type']

    systemml_args_dict = {}

    if args['stats'] is not None:
        systemml_args_dict['-stats'] = args['stats']
    else:
        systemml_args_dict['-stats'] = ''

    if args['explain'] is not None:
        systemml_args_dict['-explain'] = args['explain']
    else:
        systemml_args_dict['-explain'] = ''

    if args['config'] is not None:
        systemml_args_dict['-config'] = args['config']

    if args['gpu'] is not None:
        if args['gpu'] == 'no_option':
            systemml_args_dict['-gpu'] = ''
        else:
            systemml_args_dict['-gpu'] = args['gpu']

    backend_args_dict = {}
    exec_type = args['exec_type']
    
    if exec_type == 'hybrid_spark':
        if args['master'] is not None:
            backend_args_dict['--master'] = args['master']

        if args['deploy_mode'] is not None:
            backend_args_dict['--deploy-mode'] = args['master']

        if args['num_executors'] is not None:
            backend_args_dict['--num-executors'] = args['num_executors']

        if args['driver_memory'] is not None:
            backend_args_dict['--driver-memory'] = args['driver_memory']

        if args['executor_cores'] is not None:
            backend_args_dict['--executor-cores'] = args['executor_cores']

        if args['conf'] is not None:
            backend_args_dict['--conf'] = ''.join(args['conf'])

    elif exec_type == 'singlenode':
        if args['heapmem'] is not None:
            backend_args_dict['-heapmem'] = args['heapmem']

    return perftest_args_dict, systemml_args_dict, backend_args_dict


def args_dict_split(all_arguments):
    """
    This functions split the super set of arguments to smaller dictionaries

    all_arguments: Dictionary
    All input arguments parsed

    return: Dictionary, Dictionary, Dictionary
    We return four dictionaries for init, script, spark arguments, singlenode arguments
    """
    args_dict = dict(list(all_arguments.items())[0:9])
    config_dict = dict(list(all_arguments.items())[9:13])
    spark_dict = dict(list(all_arguments.items())[13:19])
    singlenode_dict = dict(list(all_arguments.items())[19:])

    return args_dict, config_dict, spark_dict, singlenode_dict


def get_families(current_algo, ml_algo):
    """
    Given current algorithm we get its families.

    current_algo  : String
    Input algorithm specified

    ml_algo : Dictionary
    key, value dictionary with family as key and algorithms as list of values

    return: List
    List of families returned
    """

    family_list = []
    for family, algos in ml_algo.items():
        if current_algo in algos:
            family_list.append(family)
    return family_list


def split_rowcol(matrix_dim):
    """
    Split the input matrix dimensions into row and columns

    matrix_dim: String
    Input concatenated string with row and column

    return: Tuple
    Row and column split based on suffix
    """

    k = str(0) * 3
    M = str(0) * 6
    replace_M = matrix_dim.replace('M', str(M))
    replace_k = replace_M.replace('k', str(k))
    row, col = replace_k.split('_')
    return row, col


def config_writer(write_path, config_obj):
    """
    Writes the dictionary as an configuration json file to the give path

    write_path: String
    Absolute path of file name to be written

    config_obj: List or Dictionary
    Can be a dictionary or a list based on the object passed
    """

    with open(write_path, 'w') as input_file:
        json.dump(config_obj, input_file, indent=4)


def config_reader(read_path):
    """
    Read json file given path

    return: List or Dictionary
    Reading the json file can give us a list if we have positional args or
    key value args for a dictionary
    """

    with open(read_path, 'r') as input_file:
        conf_file = json.load(input_file)

    return conf_file


def exec_dml_and_parse_time(exec_type, dml_file_name, args, backend_args_dict, systemml_args_dict, log_file_name=None):
    """
    This function is responsible of execution of input arguments via python sub process,
    We also extract time obtained from the output of this subprocess

    exec_type: String
    Contains the execution type singlenode / hybrid_spark

    dml_file_name: String
    DML file name to be used while processing the arguments give

    args: Dictionary
    Key values pairs depending on the arg type

    backend_args_dict: Dictionary
    Spark configuration arguments / singlenode config arguments

    systemml_args_dict: Dictionary
    Supplementary arguments required by the script

    log_file_name: String
    Path to write the logfile

    return: String
    The value of time parsed from the logs / error
    """

    algorithm = dml_file_name + '.dml'
    sup_args = ' '.join(['{} {}'.format(k, v) for k, v in systemml_args_dict.items()])
    if exec_type == 'singlenode':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-standalone.py')
        singlenode_pre_args = ' '.join(['{} {}'.format(k, v) for k, v in backend_args_dict.items()])
        args = ' '.join(['{} {}'.format(k, v) for k, v in args.items()])
        cmd = [exec_script, singlenode_pre_args, '-f', algorithm, args, sup_args]
        cmd_string = ' '.join(cmd)

    if exec_type == 'hybrid_spark':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-spark-submit.py')
        spark_pre_args = ' '.join([' {} {} '.format(k, v) for k, v in backend_args_dict.items()])
        args = ' '.join(['{} {}'.format(k, v) for k, v in args.items()])
        cmd = [exec_script, spark_pre_args, '-f', algorithm, args, sup_args]
        cmd_string = ' '.join(cmd)

    time = subprocess_exec(cmd_string, log_file_name, 'time')

    return time


def parse_time(raw_logs):
    """
    Parses raw input list and extracts time

    raw_logs : List
    Each line obtained from the standard output is in the list

    return: String
    Extracted time in seconds or time_not_found
    """
    # Debug
    # print(raw_logs)

    for line in raw_logs:
        if line.startswith('Total execution time'):
            extract_time = re.findall(r'\d+', line)
            total_time = '.'.join(extract_time)

            return total_time

    return 'time_not_found'


def exec_test_data(exec_type, backend_args_dict, systemml_args_dict, datagen_path, config):
    """
    Creates the test data split from the given input path

    exec_type : String
    Contains the execution type singlenode / hybrid_spark

    path : String
    Location of the input folder to pick X and Y
    """
    systemml_home = os.environ.get('SYSTEMML_HOME')
    test_split_script = join(systemml_home, 'scripts', 'perftest', 'extractTestData')
    path = join(datagen_path, config.split('/')[-1])
    X = join(path, 'X.data')
    Y = join(path, 'Y.data')
    X_test = join(path, 'X_test.data')
    Y_test = join(path, 'Y_test.data')
    args = {'-args': ' '.join([X, Y, X_test, Y_test, 'csv'])}
    exec_dml_and_parse_time(exec_type, test_split_script, args, backend_args_dict, systemml_args_dict)


def check_predict(current_algo, ml_predict):
    """
    To check if the current algorithm requires to run the predict

    current_algo: String
    Algorithm being processed

    ml_predict: Dictionary
    Key value pairs of algorithm and predict file to process
    """
    if current_algo in ml_predict.keys():
        return True


def get_folder_metrics(folder_name, action_mode):
    """
    Gets metrics from folder name for logging

    folder_name: String
    Folder from which we want to grab details

    return: List(3)
    A list with mat_type, mat_shape, intercept
    """

    if action_mode == 'data-gen':
        split_name = folder_name.split('.')
        mat_type = split_name[1]
        mat_shape = split_name[2]
        intercept = 'none'

    try:
        if action_mode == 'train':
            split_name = folder_name.split('.')
            mat_type = split_name[3]
            mat_shape = split_name[2]
            intercept = split_name[4]

        if action_mode == 'predict':
            split_name = folder_name.split('.')
            mat_type = split_name[3]
            mat_shape = split_name[2]
            intercept = split_name[4]
    except IndexError:
        intercept = 'none'

    return mat_type, mat_shape, intercept


def mat_type_check(current_family, matrix_types, dense_algos):
    """
    Some Algorithms support different matrix_types. This function give us the right matrix_type given
    an algorithm

    current_family: String
    Current family being porcessed in this function

    matrix_type: List
    Type of matrix to generate dense, sparse, all

    dense_algos: List
    Algorithms that support only dense matrix type

    return: List
    Return the list of right matrix types supported by the family
    """
    current_type = []
    for current_matrix_type in matrix_types:
        if current_matrix_type == 'all':
            if current_family in dense_algos:
                current_type.append('dense')
            else:
                current_type.append('dense')
                current_type.append('sparse')

        if current_matrix_type == 'sparse':
            if current_family in dense_algos:
                sys.exit('{} does not support {} matrix type'.format(current_family,
                                                                     current_matrix_type))
            else:
                current_type.append(current_matrix_type)

        if current_matrix_type == 'dense':
            current_type.append(current_matrix_type)

    return current_type


def get_default_dir(file_system_type, temp_dir, exec_mode, config_dir):
    """
    file_system_type: String
    temp_dir: String
    exec_mode: String
    config_dir: String

    return: String
    Local or HDFS home directory
    """

    if exec_mode == 'singlenode':
        if temp_dir is None:
            return config_dir
        if temp_dir is not None:
            return temp_dir

    if exec_mode == 'hybrid_spark':
        if file_system_type == 'hdfs':
            cmd = ['hdfs', 'getconf', '-confKey', 'fs.default.name']
            hdfs_base = subprocess_exec(' '.join(cmd), extract='hdfs_base')

            if temp_dir is None:
                hdfs_home = join(hdfs_base, 'user', getpass.getuser())
                check_hdfs_path(hdfs_home)
                return hdfs_home

            if temp_dir is not None:
                if temp_dir.startswith('hdfs'):
                    return temp_dir
                else:
                    hdfs_home = join(hdfs_base, 'user', getpass.getuser(), temp_dir)
                    return hdfs_home
        else:
            if temp_dir is None:
                return config_dir
            if temp_dir is not None:
                return temp_dir
