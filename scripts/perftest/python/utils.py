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
import sys
import glob
from functools import reduce

# This file contains all the utility functions required for performance test module


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
    key value for a dictionary
    """

    with open(read_path, 'r') as input_file:
        conf_file = json.load(input_file)

    return conf_file


def create_dir(directory):
    """
    Create directory given path if the directory does not exist already

    directory: String
    Input folder path
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_existence(path, action_mode):
    """
    Check SUCCESS file is present in the input path

    path: String
    Input folder path

    action_mode : String
    Type of action data-gen, train ...

    return: Boolean check if the file _SUCCESS exists
    """

    if action_mode == 'data-gen':
        full_path = join(path, '_SUCCESS')
        exist = os.path.isfile(full_path)
    else:
        # Files does not exist for other modes return False to continue
        # For e.g some predict algorithms do not generate an output folder
        # hence checking for SUCCESS would fail
        exist = False

    return exist


def exec_dml_and_parse_time(exec_type, dml_file_name, execution_output_file, args, time=True):
    """
    This function is responsible of execution of input arguments via python sub process,
    We also extract time obtained from the output of this subprocess

    exec_type: String
    Contains the execution type singlenode / hybrid_spark

    dml_file_name: String
    DML file name to be used while processing the arguments give

    execution_output_file: String
    Name of the file where the output of the DML run is written out

    args: Dictionary
    Key values pairs depending on the arg type

    time: Boolean (default=True)
    Boolean argument used to extract time from raw output logs.
    """

    algorithm = dml_file_name + '.dml'
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

        _, err1 = proc1.communicate()

        if "Error" in str(err1):
            print('Error Found in {}'.format(dml_file_name))
            total_time = 'failure'
        else:
            total_time = parse_time(proc1_log)

        with open(execution_output_file, 'w') as file:
            for row in proc1_log:
                file.write("%s\n" % str(row))

    else:
        total_time = 'not_specified'

    return total_time


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


def exec_test_data(exec_type, path):
    """
    Creates the test data split from the given input path

    exec_type : String
    Contains the execution type singlenode / hybrid_spark

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

    # Call the exec script without time
    config_file_name = path.split('/')[-1]
    exec_dml_and_parse_time(exec_type, test_split_script, config_file_name, args, False)


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
    Gets metrics from folder name

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
    Some Algorithms support different matrix_type. This function give us the right matrix_type given
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


def relevant_folders(path, algo, family, matrix_type, matrix_shape, mode):
    """
    Finds the right folder to read the data based on given parameters

    path: String
    Location of data-gen and training folders

    algo: String
    Current algorithm being processed by this function

    family: String
    Current family being processed by this function

    matrix_type: List
    Type of matrix to generate dense, sparse, all

    matrix_shape: List
    Dimensions of the input matrix with rows and columns

    mode: String
    Based on mode and arguments we read the specific folders e.g data-gen folder or train folder

    return: List
    List of folder locations to read data from
    """
    folders = []
    for current_matrix_type in matrix_type:
        for current_matrix_shape in matrix_shape:
            if mode == 'data-gen':
                data_gen_path = join(path, family)
                sub_folder_name = '.'.join([current_matrix_type, current_matrix_shape])
                path_subdir = glob.glob(data_gen_path + '.' + sub_folder_name + "*")

            if mode == 'train':
                train_path = join(path, algo)
                sub_folder_name = '.'.join([family, current_matrix_type, current_matrix_shape])
                path_subdir = glob.glob(train_path + '.' + sub_folder_name + "*")

            path_folders = list(filter(lambda x: os.path.isdir(x), path_subdir))
            folders.append(path_folders)

    folders_flat = reduce(lambda x, y: x + y, folders)

    return folders_flat
