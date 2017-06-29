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
from utils import config_writer
from functools import reduce

# Contains configuration setting for training
DATA_FORMAT = 'csv'


def binomial_m_svm_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []
    for i in [0, 1]:
        icpt = str(i)
        reg = '0.01'
        tol = '0.0001'
        maxiter = 20
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        data_folders.append(full_path_train)

        model = join(full_path_train, 'model.data')
        Log = join(full_path_train, 'Log.data')

        config = dict(X=X, Y=Y, icpt=icpt, classes=2, reg=reg, tol=tol, maxiter=maxiter, model=model,
                      Log=Log, fmt=DATA_FORMAT)
        config_writer(full_path_train + '.json', config)

    return data_folders


def binomial_l2_svm_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []
    for i in [0, 1]:
        icpt = str(i)
        reg = '0.01'
        tol = '0.0001'
        maxiter = '100'
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        data_folders.append(full_path_train)

        model = join(full_path_train, 'model.data')
        Log = join(full_path_train, 'Log.data')

        config = dict(X=X, Y=Y, icpt=icpt, reg=reg, tol=tol, maxiter=maxiter, model=model,
                      Log=Log, fmt=DATA_FORMAT)
        config_writer(full_path_train + '.json', config)

    return data_folders


def binomial_multilogreg_train(save_folder_name, datagen_dir, train_dir):
    data_folders = []

    for i in [0, 1, 2]:
        icpt = str(i)
        reg = '0.01'
        tol = '0.0001'
        moi = '100'
        mii = '5'
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        data_folders.append(full_path_train)

        B = join(full_path_train, 'B.data')

        config = dict(X=X, Y=Y, icpt=icpt, reg=reg, tol=tol, moi=moi, mii=mii,
                      B=B)
        config_writer(full_path_train + '.json', config)
    return data_folders


def multinomial_m_svm_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []
    for i in [0, 1]:
        icpt = str(i)
        reg = '0.01'
        tol = '0.0001'
        maxiter = '20'
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        model = join(full_path_train, 'model.data')
        Log = join(full_path_train, 'Log.data')

        config = dict(X=X, Y=Y, icpt=icpt, classes=150, reg=reg, tol=tol, maxiter=maxiter, model=model,
                      Log=Log, fmt=DATA_FORMAT)
        config_writer(full_path_train + '.json', config)
        data_folders.append(full_path_train)

    return data_folders


def clustering_kmeans_train(save_folder_name, datagen_dir, train_dir):

    X = join(datagen_dir, 'X.data')

    full_path_train = join(train_dir, save_folder_name)
    C = join(full_path_train, 'C.data')
    k = '50'
    maxi = '50'
    tol = '0.0001'
    config = dict(X=X, k=k, maxi=maxi, tol=tol, C=C)

    config_writer(full_path_train + '.json', config)

    return [full_path_train]


def stats1_univar_stats_train(save_folder_name, datagen_dir, train_dir):

    X = join(datagen_dir, 'X.data')
    TYPES = join(datagen_dir, 'types')

    full_path_train = join(train_dir, save_folder_name)
    STATS = join(full_path_train, 'STATS.data')

    config = dict(X=X, TYPES=TYPES, STATS=STATS)
    config_writer(full_path_train + '.json', config)

    return [full_path_train]


def stats1_bivar_stats_train(save_folder_name, datagen_dir, train_dir):

    X = join(datagen_dir, 'X.data')
    index1 = join(datagen_dir, 'set1.indices')
    index2 = join(datagen_dir, 'set2.indices')
    types1 = join(datagen_dir, 'set1.types')
    types2 = join(datagen_dir, 'set2.types')

    full_path_train = join(train_dir, save_folder_name)
    OUTDIR = full_path_train

    config = dict(X=X, index1=index1, index2=index2, types1=types1, types2=types2, OUTDIR=OUTDIR)
    config_writer(full_path_train + '.json', config)
    return [full_path_train]


def stats2_stratstats_train(save_folder_name, datagen_dir, train_dir):

    X = join(datagen_dir, 'X.data')
    Xcid = join(datagen_dir, 'Xcid.data')
    Ycid = join(datagen_dir, 'Ycid.data')

    full_path_train = join(train_dir, save_folder_name)
    O = join(full_path_train, 'O.data')

    config = dict(X=X, Xcid=Xcid, Ycid=Ycid, O=O, fmt=DATA_FORMAT)

    config_writer(full_path_train + '.json', config)

    return [full_path_train]


def multinomial_naive_bayes_train(save_folder_name, datagen_dir, train_dir):

    X = join(datagen_dir, 'X.data')
    Y = join(datagen_dir, 'Y.data')
    classes = '150'

    full_path_train = join(train_dir, save_folder_name)
    prior = join(full_path_train, 'prior')
    conditionals = join(full_path_train, 'conditionals')
    accuracy = join(full_path_train, 'accuracy')
    fmt = DATA_FORMAT
    probabilities = join(full_path_train, 'probabilities')

    config = dict(X=X, Y=Y, classes=classes, prior=prior, conditionals=conditionals,
                  accuracy=accuracy, fmt=fmt, probabilities=probabilities)

    config_writer(full_path_train + '.json', config)

    return [full_path_train]


def multinomial_multilogreg_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []
    for i in [0, 1, 2]:
        icpt = str(i)
        reg = '0.01'
        tol = '0.0001'
        moi = '100'
        mii = '0'
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        data_folders.append(full_path_train)
        B = join(full_path_train, 'B.data')

        config = dict(X=X, Y=Y, B=B, icpt=icpt, reg=reg, tol=tol, moi=moi, mii=mii, fmt=DATA_FORMAT)
        config_writer(full_path_train + '.json', config)

    return data_folders


def regression1_linearregds_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []
    for i in [0, 1, 2]:
        icpt = str(i)
        reg = '0.01'
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        data_folders.append(full_path_train)
        B = join(full_path_train, 'B.data')

        config = dict(X=X, Y=Y, B=B, icpt=icpt, fmt=DATA_FORMAT, reg=reg)
        config_writer(full_path_train + '.json', config)

    return data_folders


def regression1_linearregcg_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []
    for i in [0, 1, 2]:
        icpt = str(i)
        reg = '0.01'
        tol = '0.0001'
        maxi = '20'
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name + '.' + str(i))
        data_folders.append(full_path_train)
        B = join(full_path_train, 'B.data')

        config = dict(X=X, Y=Y, B=B, icpt=icpt, fmt=DATA_FORMAT, maxi=maxi, tol=tol, reg=reg)
        config_writer(full_path_train + '.json', config)

    return data_folders


def regression2_glm_gamma_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []

    for i in [0, 1, 2]:
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name)
        data_folders.append(full_path_train)

        B = join(full_path_train, 'B.data')
        icpt = str(i)
        fmt = DATA_FORMAT
        moi = '200'
        mii = '5'
        dfam = '1'
        vpow = '2.0'
        link = '1'
        lpow = '0.0'
        tol = '0.0001'
        reg = '0.01'
        config = dict(X=X, Y=Y, B=B, icpt=icpt, fmt=fmt, moi=moi, mii=mii, dfam=dfam,
                      vpov=vpow, link=link, lpow=lpow, tol=tol, reg=reg)

        config_writer(full_path_train + '.json', config)

    return data_folders


def regression2_glm_binomial_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []

    for i in [0, 1, 2]:
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name)
        data_folders.append(full_path_train)

        B = join(full_path_train, 'B.data')
        icpt = str(i)
        fmt = DATA_FORMAT
        moi = '200'
        mii = '5'
        dfam = '2'
        link = '3'
        yneg = '2'
        tol = '0.0001'
        reg = '0.01'
        config = dict(X=X, Y=Y, B=B, icpt=icpt, fmt=fmt, moi=moi, mii=mii,
                      dfam=dfam, link=link, yneg=yneg, tol=tol, reg=reg)

        config_writer(full_path_train + '.json', config)

    return data_folders


def regression2_glm_poisson_train(save_folder_name, datagen_dir, train_dir):

    data_folders = []

    for i in [0, 1, 2]:
        X = join(datagen_dir, 'X.data')
        Y = join(datagen_dir, 'Y.data')

        full_path_train = join(train_dir, save_folder_name)
        data_folders.append(full_path_train)

        B = join(full_path_train, 'B.data')
        icpt = str(i)
        fmt = DATA_FORMAT
        moi = '200'
        mii = '5'
        dfam = '1'
        vpov = '1'
        link = '1'
        lpow = '0'
        tol = '0.0001'
        reg = '0.01'
        config = dict(X=X, Y=Y, B=B, icpt=icpt, fmt=fmt, moi=moi, mii=mii,
                      dfam=dfam, vpov=vpov, link=link, lpow=lpow, tol=tol, reg=reg)
        config_writer(full_path_train + '.json', config)

    return data_folders


def config_packets_train(algo_payload, datagen_dir, train_dir):
    """
    This function has two responsibilities. Generate the configuration files for
    input training algorithms and return a dictionary that will be used for execution.

    algo_payload : List of tuples
    The first tuple index contains algorithm name and the second index contains
    family type.

    datagen_dir: String
    Path of the data generation directory

    train_dir: String
    Path of the training directory

    return: {string: list}
    This dictionary contains algorithms to be executed as keys and the path of configuration
    json files to be executed list of values.

    """

    config_bundle = {}

    for k, v in algo_payload:
        config_bundle[k] = []

    for current_algo, current_family in algo_payload:
        data_gen_path = join(datagen_dir, current_family)
        data_gen_subdir = glob.glob(data_gen_path + "*")

        # Filter for specific data gen
        data_gen_folders = list(filter(lambda x: os.path.isdir(x), data_gen_subdir))
        if len(data_gen_folders) == 0:
            print('datagen folders not present for {}'.format(current_family))
            sys.exit()

        for current_folder in data_gen_folders:
            file_path_last = current_folder.split('/')[-1]
            save_name = '.'.join([current_algo] + [file_path_last])
            algo_func = '_'.join([current_family] + [current_algo.lower().replace('-', '_')]
                                 + ['train'])
            conf_path = globals()[algo_func](save_name, current_folder, train_dir)
            config_bundle[current_algo].append(conf_path)

    config_packets = {}

    # Flatten
    for current_algo, current_family in config_bundle.items():
        config_packets[current_algo] = reduce(lambda x, y: x + y, current_family)

    return config_packets
