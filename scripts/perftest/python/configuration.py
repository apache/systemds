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
from utils import split_rowcol, config_writer
import sys
import logging

mat_type = {'dense': 0.9,
            'sparse': 0.01}
format = 'csv'
has_predict = ['GLM', 'Kmeans', 'l2-svm', 'm-svm', 'naive-bayes']


def naive_bayes_datagen(matrix_type, mat_shapes, conf_dir):
    for index, dim in enumerate(mat_shapes):
        file_name = '_'.join(['naive_bayes_datagen', matrix_type, str(index) + '.json'])

        config = [dim[0], dim[1], mat_type[matrix_type], 150, 0,
                  'X.data', 'Y.data', format]
        config_writer(conf_dir, config, file_name)
    return None


def naive_bayes_datagen(matrix_type, mat_shapes, conf_dir):
    for index, dim in enumerate(mat_shapes):
        file_name = '_'.join(['naive_bayes_datagen', matrix_type, str(index) + '.json'])

        config = [dim[0], dim[1], mat_type[matrix_type], 150, 0,
                  'X.data', 'Y.data', format]
        config_writer(conf_dir, config, file_name)
    return None


def kmeans_datagen(matrix_type, mat_shapes, conf_dir):
    for index, dim in enumerate(mat_shapes):
        file_name = '_'.join(['kmeans_datagen', str(index) + '.json'])
        config = dict(nr=dim[0], nf=dim[1], nc='5', dc='10.0', dr='1.0',
                      fbf='100.0', cbf='100.0', X='X.data', C='C.data', Y='Y.data',
                      YbyC='YbyC.data', fmt=format)
        config_writer(conf_dir, config, file_name)
    return None


def kmeans_train(conf_dir):
    file_name = ''.join(['kmeans_train', '.json'])
    config = dict(X='X.data', k=5, maxi=10, runs=10, tol=0.00000001, samp=20,
                  C='C.data', isY='TRUE', Y='Y.data', verb='TRUE')
    config_writer(conf_dir, config, file_name)
    return None


def kmeans_predict(conf_dir):
    file_name = ''.join(['kmeans_predict', '.json'])
    config = dict(X='X.data', C='C.data', prY='prY.data')
    config_writer(conf_dir, config, file_name)
    return None


def init_conf(algo, temp_dir, matrix_type, matrix_shape, job):
    # Create directories
    conf_dir = join(temp_dir, 'conf')
    gen_dir = join(temp_dir, 'data_gen')
    train_dir = join(temp_dir, 'train')
    pred_dir = join(temp_dir, 'pred')

    for dirs in [conf_dir, gen_dir, train_dir, pred_dir]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    mat_shapes = split_rowcol(matrix_shape)

    if job[0] == 1:
        for current_algo in algo:
            algo_dg = current_algo.lower().replace('-', '_') + '_datagen'
            globals()[algo_dg](matrix_type, mat_shapes, conf_dir)
            logging.info('Completed writing {} datagen file'.format(current_algo))

    if job[1] == 1:
        for current_algo in algo:
            algo_dg = current_algo.lower() + '_train'
            globals()[algo_dg](conf_dir)
            logging.info('Completed writing {} training file'.format(current_algo))

    if job[2] == 1:
        for current_algo in algo:
            if current_algo in has_predict:
                algo_dg = current_algo.lower() + '_predict'
                globals()[algo_dg](conf_dir)
                logging.info('Completed writing {} training file'.format(current_algo))
