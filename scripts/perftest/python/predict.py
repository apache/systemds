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
import glob
from utils import create_dir, config_writer

# Contains configuration setting for predicting

HAS_PREDICT = ['Kmeans']
DATA_FORMAT = 'csv'


def kmeans_predict(save_file_name, load_datagen, load_train, datagen_dir, train_dir, predict_dir):

    full_path_datagen = join(datagen_dir, load_datagen)

    X = join(full_path_datagen, 'X.data')
    C = join(full_path_datagen, 'C.data')

    full_path_predict = join(predict_dir, save_file_name)
    prY = join(full_path_predict, 'prY.data')

    config = dict(X=X, C=C, prY=prY)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def config_packets_predict(algo_payload, datagen_dir, train_dir, predict_dir):

    config_bundle = {}
    for current_algo, current_family in algo_payload.items():
        if current_algo in HAS_PREDICT:

            # Find datagen folders
            data_gen_path = join(datagen_dir, current_family)
            data_gen_subdir = glob.glob(data_gen_path + "*")
            data_gen_folders = filter(lambda x: os.path.isdir(x), data_gen_subdir)

            # Find train folders
            train_path = join(train_dir, current_algo)
            train_subdir = glob.glob(train_path + "*")
            train_folders = filter(lambda x: os.path.isdir(x), train_subdir)
            config_bundle[current_algo] = []

            for data_gen, train in zip(data_gen_folders, train_folders):
                algo_func = current_algo.lower() + '_predict'
                load_datagen = data_gen.split('/')[-1]
                load_train = train.split('/')[-1]
                create_dir(predict_dir + os.sep + load_train)

                conf_path = globals()[algo_func](load_train, load_datagen, load_train,
                                                 datagen_dir, train_dir, predict_dir)
                config_bundle[current_algo].append(conf_path)

    return config_bundle
