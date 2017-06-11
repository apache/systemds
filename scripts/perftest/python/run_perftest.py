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

# TODO:
# Logging
# Handel Intercept

import sys
import argparse
from functools import reduce
import os
from os.path import join

ml_algo = {'binomial': ['MultiLogReg', 'l2-svm', 'm-svm'],
           'clustering': ['Kmeans'],
           'multinomial': ['naive-bayes', 'MultiLogReg', 'm-svm'],
           'regression': ['LinearRegDS', 'LinearRegCG', 'GLM'],
           'stats': ['Univar-Stats', 'bivar-stats', 'stratstats']}

has_predict = ['GLM', 'Kmeans', 'l2-svm', 'm-svm', 'naive-bayes']


def kmeans():

    pass


def gen_config(algo, conf_dir, matrix_type, rows, cols):
    # Create Configuration for
    # data gen
    # train
    # predict
    algo_low = algo.lower()
    globals()[algo_low]()


    pass


def split_rowcol(matrix_dim):
    k = str(0) * 3
    M = str(0) * 6
    replace_M = matrix_dim.replace('M', str(M))
    replace_k = replace_M.replace('k', str(k))
    row, col = replace_k.split('_')
    return row, col


def init_conf(algo, temp_dir, matrix_type, matrix_shape):

    # Create directories
    conf_dir = join(temp_dir, 'conf')
    gen_dir = join(temp_dir, 'data_gen')
    train_dir = join(temp_dir, 'train')
    pred_dir = join(temp_dir, 'pred')

    for dirs in [conf_dir, gen_dir, train_dir, pred_dir]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    row, col = split_rowcol(matrix_shape)

    for current_algo in algo:
        gen_config(current_algo, conf_dir, matrix_type, row, col)
        break


def data_gen(algo, exec_type):
    # If init directory not presnt break
    # Reads configuration files from the config directory
    # Generates the data in datagen directory
    # do not write if data already present
    # METRICS : Time, Capture Error
    pass


def train(algo, exec_type):
    # if data_gen directory not present break
    # reads data from data_gen directory
    # writes output to train directory
    # METRICS : Time, Capture Error
    pass


def predict(algo, exec_type):
    # train driectory not present break
    # read from training directory
    # write output to predict dir
    # METRICS : Time, Capture Error
    pass


def family_algo(family):
    algo = []
    for fam in family:
        algo.append(ml_algo[fam])
    algo_flat = reduce(lambda x, y: x + y, algo)
    return algo_flat


def main(family, algo, exec_type, matrix_type, matrix_shape, temp_dir, init, generate_data, train, predict):

    if algo is None:
        algo = family_algo(family)

    if init:
        init_conf(algo, temp_dir, matrix_type, matrix_shape)

    if generate_data:
        data_gen(algo, exec_type)

    if train:
        train(algo, exec_type)

    if predict:
        predict(algo, exec_type)

    pass


if __name__ == '__main__':
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')

    group = cparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--family', help='specify class of algorithms (e.g regression, binomial, all)',
                       metavar='', nargs='+')
    group.add_argument('--algo', help='specify the type of algorithm to run', metavar='', nargs='+')
    cparser.add_argument('-exec-type', default='singlenode', help='System-ML backend (e.g singlenode, '
                                                                  'spark, spark-hybrid)', metavar='')
    cparser.add_argument('--matrix-type', required=True, help='Type of matrix to generate (e.g dense '
                                                              'or sparse)', metavar=None)
    cparser.add_argument('--matrix-shape', required=True, help='Shape of matrix to generate (e.g '
                                                               '10k_1k)', metavar=None)

    # Optional Arguments
    cparser.add_argument('-temp-dir', help='specify temporary directory', metavar='')
    cparser.add_argument('--init', help='generate configuration files', action='store_true')
    cparser.add_argument('--generate-data', help='generate data', action='store_true')
    cparser.add_argument('--train', help='train algorithms', action='store_true')
    cparser.add_argument('--predict', help='predict (if available)', action='store_true')

    args = cparser.parse_args()
    arg_dict = vars(args)

    # Check for validity of input arguments
    if args.family is not None:
        for fam in args.family:
            if fam not in ml_algo.keys():
                print('{} family not present in the performance test suit'.format(fam))
                sys.exit()

    if args.algo is not None:
        for algo in args.algo:
            algo_flat = reduce(lambda x, y: x+y, ml_algo.values())
            if algo not in algo_flat:
                print('{} algorithm not present in the performance test suit'.format(args.algo))
                sys.exit()

    if args.temp_dir is None:
        systemml_home = os.environ.get('SYSTEMML_HOME')
        args.temp_dir = join(systemml_home, 'scripts', 'perftest', 'temp')

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    main(**arg_dict)