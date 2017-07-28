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
import os
import json
import subprocess
import shlex
import re
import logging
import sys
import glob
from functools import reduce
from execution import subprocess_exec


def create_dir_local(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_success(time, cwd):
    if 'data-gen' in cwd:
        if cwd.startswith('hdfs') and len(time.split('.')) == 2:
            full_path = join(cwd, '_SUCCESS')
            cmd = ['hdfs', 'dfs', '-touchz', full_path]
            subprocess_exec(' '.join(cmd))
        else:
            if len(time.split('.')) == 2:
                full_path = join(cwd, '_SUCCESS')
                open(full_path, 'w').close()


def get_existence(path):
    """
    Check SUCCESS file is present in the input path

    path: String
    Input folder path

    action_mode : String
    Type of action data-gen, train ...

    return: Boolean check if the file _SUCCESS exists
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


def relevant_folders_local(path, algo, family, matrix_type, matrix_shape, mode):
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


def relevant_folders_hdfs(path, algo, family, matrix_type, matrix_shape, mode):

    folders = []
    for current_matrix_type in matrix_type:
        for current_matrix_shape in matrix_shape:
            if mode == 'data-gen':
                sub_folder_name = '.'.join([family, current_matrix_type, current_matrix_shape])
                cmd = ['hdfs', 'dfs', '-ls', path]
                path_subdir = subprocess_exec(' '.join(cmd), 'dir')

            if mode == 'train':
                sub_folder_name = '.'.join([algo, family, current_matrix_type, current_matrix_shape])
                cmd = ['hdfs', 'dfs', '-ls', path]
                path_subdir = subprocess_exec(' '.join(cmd), 'dir')

            path_folders = list(filter(lambda x: contains_dir(x, sub_folder_name), path_subdir))
            folders.append(path_folders)

    folders_flat = reduce(lambda x, y: x + y, folders)

    return folders_flat


def contains_dir(hdfs_dirs, sub_folder):
    if sub_folder in hdfs_dirs:
        return True
    else:
        # Debug
        # print('{}, {}'.format(sub_folder, hdfs_dirs))
        pass
    return False