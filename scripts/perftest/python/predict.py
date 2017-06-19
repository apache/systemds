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
import sys
import glob
from utils import create_dir, config_writer

# Contains configuration setting for predicting

has_predict = ['Kmeans']
format = 'csv'


def kmeans_predict(file_name, datagen_dir, train_dir, predict_dir):

    full_path_datagen = join(datagen_dir, file_name)

    X = join(full_path_datagen, 'X.data')
    C = join(full_path_datagen, 'C.data')

    full_path_predict = join(predict_dir, file_name)
    prY = join(full_path_predict, 'prY.data')

    config = dict(X=X, C=C, prY=prY)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def multilogreg_predict(file_name, datagen_dir, train_dir, predict_dir):
    # TODO:
    # Incomplete

    full_path_train = join(train_dir, file_name)

    dfam = 2
    C = join(full_path_datagen, 'C.data')
    X = join(full_path_datagen, 'X.data')
    B = join(full_path_datagen, 'B.data')
    Y = join(full_path_datagen, 'Y.data')
    M = join(full_path_datagen, 'M.data')

    full_path_predict = join(predict_dir, file_name)
    O = join(full_path_predict, 'O.data')

    config = dict(dfam=dfam, vpow=1, link=2, lpow=-1, fmt=format, )
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def config_packets_predict(algos, datagen_dir, train_dir, predict_dir):

    config_bundle = {}

    for current_algo in algos:
        if current_algo in has_predict:
            datagen_path = datagen_dir + os.sep + current_algo
            datagen_subdir = glob.glob(datagen_path + "*")
            datagen_folders = filter(lambda x: os.path.isdir(x), datagen_subdir)
            config_bundle[current_algo] = []

            for folder in datagen_folders:
                algo_func = current_algo.lower() + '_predict'
                file_name = folder.split('/')[-1]
                create_dir(predict_dir + os.sep + file_name)
                conf_path = globals()[algo_func](file_name, datagen_dir, train_dir, predict_dir)
                config_bundle[current_algo].append(conf_path)

    return config_bundle
