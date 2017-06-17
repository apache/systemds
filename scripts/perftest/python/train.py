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

import sys
import glob
import os
from os.path import join
from utils import split_rowcol, config_writer, create_dir

# Contains configuration setting for training


def kmeans_train(file_name, datagen_dir, train_dir):

    full_path_datagen = join(datagen_dir, file_name)
    X = join(full_path_datagen, 'X.data')

    full_path_train = join(train_dir, file_name)
    C = join(full_path_train, 'C.data')

    config = dict(X=X, k='50', maxi='50', tol='0.0001',
                  C=C)

    config_writer(full_path_train + '.json', config)

    return full_path_train


def config_packets_train(algos, datagen_dir, train_dir):

    config_bundle = {}

    # create ini in train dir
    for current_algo in algos:
        datagen_path = datagen_dir + os.sep + current_algo
        datagen_subdir = glob.glob(datagen_path + "*")
        datagen_folders = filter(lambda x: os.path.isdir(x), datagen_subdir)
        config_bundle[current_algo] = []

        for folder in datagen_folders:
            algo_func = current_algo.lower() + '_train'
            file_name = folder.split('/')[-1]
            create_dir(train_dir + os.sep + file_name)
            conf_path = globals()[algo_func](file_name, datagen_dir, train_dir)
            config_bundle[current_algo].append(conf_path)

    return config_bundle