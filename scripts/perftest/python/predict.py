#!/usr/bin/env python3
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for dadditional information
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
from os.path import join
from utils import config_writer, relevant_folders, mat_type_check

# Contains configuration setting for predicting
DATA_FORMAT = 'csv'


def m_svm_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    X = join(datagen_dir, 'X_test.data')
    Y = join(datagen_dir, 'Y_test.data')

    icpt = save_file_name.split('.')[-1]
    model = join(train_dir, 'model.data')
    fmt = DATA_FORMAT

    config = dict(X=X, Y=Y, icpt=icpt, model=model, fmt=fmt)

    full_path_predict = join(predict_dir, save_file_name)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def l2_svm_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    X = join(datagen_dir, 'X_test.data')
    Y = join(datagen_dir, 'Y_test.data')

    icpt = save_file_name.split('.')[-1]
    model = join(train_dir, 'model.data')
    fmt = DATA_FORMAT

    config = dict(X=X, Y=Y, icpt=icpt, model=model, fmt=fmt)

    full_path_predict = join(predict_dir, save_file_name)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def multilogreg_predict(save_file_name, datagen_dir, train_dir, predict_dir):
    X = join(datagen_dir, 'X_test.data')
    Y = join(datagen_dir, 'Y_test.data')
    B = join(train_dir, 'B.data')
    M = join(train_dir, 'M.data')
    dfam = '3'
    vpow = '-1'
    link = '2'
    fmt = DATA_FORMAT

    config = dict(dfam=dfam, vpow=vpow, link=link, fmt=fmt, X=X, B=B, Y=Y, M=M)

    full_path_predict = join(predict_dir, save_file_name)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def naive_bayes_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    X = join(datagen_dir, 'X_test.data')
    Y = join(datagen_dir, 'Y_test.data')

    prior = join(train_dir, 'prior')
    conditionals = join(train_dir, 'conditionals')
    fmt = DATA_FORMAT
    probabilities = join(train_dir, 'probabilities')
    config = dict(X=X, Y=Y, prior=prior, conditionals=conditionals, fmt=fmt, probabilities=probabilities)

    full_path_predict = join(predict_dir, save_file_name)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def kmeans_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    X = join(datagen_dir, 'X_test.data')
    C = join(datagen_dir, 'C.data')

    full_path_predict = join(predict_dir, save_file_name)
    prY = join(full_path_predict, 'prY.data')

    config = dict(X=X, C=C, prY=prY)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def linearregcg_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    dfam = '1'
    link = '1'
    vpow = '0.0'
    lpow = '1.0'

    X = join(datagen_dir, 'X_test.data')
    B = join(train_dir, 'B.data')
    Y = join(datagen_dir, 'Y_test.data')

    full_path_predict = join(predict_dir, save_file_name)
    M = join(full_path_predict, 'M.data')
    O = join(full_path_predict, 'O.data')

    config = dict(dfam=dfam, link=link, vpow=vpow, lpow=lpow, fmt=DATA_FORMAT, X=X,
                  B=B, Y=Y, M=M, O=O)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def linearregds_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    dfam = '1'
    link = '1'
    vpow = '0.0'
    lpow = '1.0'

    X = join(datagen_dir, 'X_test.data')
    B = join(train_dir, 'B.data')
    Y = join(datagen_dir, 'Y_test.data')

    full_path_predict = join(predict_dir, save_file_name)
    M = join(full_path_predict, 'M.data')
    O = join(full_path_predict, 'O.data')

    config = dict(dfam=dfam, link=link, vpow=vpow, lpow=lpow, fmt=DATA_FORMAT, X=X,
                  B=B, Y=Y, M=M, O=O)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def glm_poisson_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    dfam = '1'
    link = '1'
    vpow = '1'
    lpow = '1.0'

    X = join(datagen_dir, 'X_test.data')
    B = join(train_dir, 'B.data')
    Y = join(datagen_dir, 'Y_test.data')

    full_path_predict = join(predict_dir, save_file_name)
    M = join(full_path_predict, 'M.data')
    O = join(full_path_predict, 'O.data')

    config = dict(dfam=dfam, link=link, vpow=vpow, lpow=lpow, fmt=DATA_FORMAT, X=X,
                  B=B, Y=Y, M=M, O=O)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def glm_binomial_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    dfam = '2'
    link = '3'

    X = join(datagen_dir, 'X_test.data')
    B = join(train_dir, 'B.data')
    Y = join(datagen_dir, 'Y_test.data')

    full_path_predict = join(predict_dir, save_file_name)
    M = join(full_path_predict, 'M.data')
    O = join(full_path_predict, 'O.data')

    config = dict(dfam=dfam, link=link, fmt=DATA_FORMAT, X=X,
                  B=B, Y=Y, M=M, O=O)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def glm_gamma_predict(save_file_name, datagen_dir, train_dir, predict_dir):

    dfam = '1'
    link = '1'
    vpow = '2'
    lpow = '0'

    X = join(datagen_dir, 'X_test.data')
    B = join(train_dir, 'B.data')
    Y = join(datagen_dir, 'Y_test.data')

    full_path_predict = join(predict_dir, save_file_name)
    M = join(full_path_predict, 'M.data')
    O = join(full_path_predict, 'O.data')

    config = dict(dfam=dfam, link=link, vpow=vpow, lpow=lpow, fmt=DATA_FORMAT, X=X,
                  B=B, Y=Y, M=M, O=O)
    config_writer(full_path_predict + '.json', config)

    return full_path_predict


def config_packets_predict(algo_payload, matrix_type, matrix_shape, datagen_dir, train_dir, predict_dir, dense_algos):
    """
    This function has two responsibilities. Generate the configuration files for
    prediction algorithms and return a dictionary that will be used for execution.

    algo_payload : List of tuples
    The first tuple index contains algorithm name and the second index contains
    family type.

    matrix_type: String
    Type of matrix to generate e.g dense, sparse, all

    matrix_shape: String
    Shape of matrix to generate e.g 100k_10

    datagen_dir: String
    Path of the data generation directory

    train_dir: String
    Path of the training directory

    predict_dir: String
    Path of the prediction directory

    dense_algos: List
    Algorithms that support only dense matrix type

    return: Dictionary  {string: list}
    This dictionary contains algorithms to be executed as keys and the path of configuration
    json files to be executed list of values.
    """
    config_bundle = {}

    for k, _ in algo_payload:
        config_bundle[k] = []

    for current_algo, current_family in algo_payload:
        current_matrix_type = mat_type_check(current_family, matrix_type, dense_algos)
        train_folders = relevant_folders(train_dir, current_algo, current_family,
                                         current_matrix_type, matrix_shape, 'train')

        if len(train_folders) == 0:
            print('training folders not present for {}'.format(current_algo))
            sys.exit()

        for current_train_folder in train_folders:
            current_data_gen_dir = relevant_folders(datagen_dir, current_algo, current_family,
                                                    current_matrix_type, matrix_shape, 'data-gen')
            if len(current_data_gen_dir) == 0:
                print('data-gen folders not present for {}'.format(current_family))
                sys.exit()

            save_name = current_train_folder.split('/')[-1]
            algo_func = '_'.join([current_algo.lower().replace('-', '_')] + ['predict'])

            # current_data_gen_dir has index 0 as we would expect one datagen for each algorithm
            conf_path = globals()[algo_func](save_name, current_data_gen_dir[0],
                                             current_train_folder, predict_dir)

            config_bundle[current_algo].append(conf_path)

    return config_bundle
