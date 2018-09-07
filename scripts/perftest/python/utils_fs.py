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
#-------------------------------------------------------------

import os
import sys
from os.path import join
import glob
from functools import reduce
from utils_exec import subprocess_exec

# Utility support for all file system related operations


def create_dir_local(directory):
    """
    Create a directory in the local fs

    directory: String
    Location to create a directory
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def write_success(time, path):
    """
    Write SUCCESS file in the given directory

    time: String
    Time taken to execute the dml script

    path: String
    Location to write the SUCCESS file
    """

    if 'data-gen' in path:
        if path.startswith('hdfs') and len(time.split('.')) == 2:
            full_path = join(path, '_SUCCESS')
            cmd = ['hdfs', 'dfs', '-touchz', full_path]
            subprocess_exec(' '.join(cmd))
        else:
            if len(time.split('.')) == 2:
                full_path = join(path, '_SUCCESS')
                open(full_path, 'w').close()


def check_SUCCESS_file_exists(path):
    """
    Check SUCCESS file is present in the input path

    path: String
    Input folder path

    action_mode : String
    Type of action data-gen, train ...

    return: Boolean
    Checks if the file _SUCCESS exists
    """

    if 'data-gen' in path:
        if path.startswith('hdfs'):
            full_path = join(path, '_SUCCESS')
            cmd = ['hdfs', 'dfs', '-test', '-e', full_path]
            return_code = os.system(' '.join(cmd))
            if return_code == 0:
                return True
        else:
            full_path = join(path, '_SUCCESS')
            exist = os.path.isfile(full_path)
            return exist
    return False


def contains_dir(hdfs_dirs, sub_folder):
    """
    Support for Lambda Function to check if a HDFS subfolder is contained by the HDFS directory
    """

    if sub_folder in hdfs_dirs:
        return True
    else:
        # Debug
        # print('{}, {}'.format(sub_folder, hdfs_dirs))
        pass
    return False


def check_hdfs_path(path):
    """
    Check if a path is present in HDFS
    """

    cmd = ['hdfs', 'dfs', '-test', '-e', path]
    return_code = subprocess_exec(' '.join(cmd))
    if return_code != 0:
        return sys.exit('Please create {}'.format(path))


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
            if path.startswith('hdfs'):
                if mode == 'data-gen':
                    sub_folder_name = '.'.join([family, current_matrix_type, current_matrix_shape])
                    cmd = ['hdfs', 'dfs', '-ls', path]
                    path_subdir = subprocess_exec(' '.join(cmd), extract='dir')

                if mode == 'train':
                    sub_folder_name = '.'.join([algo, family, current_matrix_type, current_matrix_shape])
                    cmd = ['hdfs', 'dfs', '-ls', path]
                    path_subdir = subprocess_exec(' '.join(cmd), extract='dir')

                path_folders = list(filter(lambda x: contains_dir(x, sub_folder_name), path_subdir))

            else:
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
