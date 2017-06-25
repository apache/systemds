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
# -------------------------------------------------------------

import sys
import time
import argparse
from functools import reduce
import os
from os.path import join
from utils import get_family, config_reader, create_dir, exec_func, get_config
import logging
from datagen import config_packets_datagen
from train import config_packets_train
from predict import config_packets_predict


ML_GENDATA = {'binomial': 'genRandData4LogisticRegression',
              'clustering': 'genRandData4Kmeans',
              'multinomial': 'genRandData4Multinomial',
              'regression1': 'genRandData4LogisticRegression',
              'regression2': 'genRandData4LogisticRegression',
              'stats1': 'genRandData4DescriptiveStats',
              'stats2': 'genRandData4StratStats'}

ML_ALGO = {'binomial': ['MultiLogReg', 'l2-svm', 'm-svm'],
           'clustering': ['Kmeans'],
           'multinomial': ['naive-bayes', 'MultiLogReg', 'm-svm'],
           'regression1': ['LinearRegDS', 'LinearRegCG'],
           'regression2': ['GLM_poisson', 'GLM_gamma', 'GLM_binomial'],
           'stats1': ['Univar-Stats', 'bivar-stats'],
           'stats2': ['stratstats']}

ML_PREDICT = {'Kmeans': 'Kmeans-predict'}


# Since m-svm appears in binomial and multinomial family
HAS_DUAL = 'm-svm'


def algorithm_workflow(algo, exec_type, config_path, file_name, type):

    config_data = config_reader(config_path + '.json')

    if isinstance(config_data, dict):
        dict_args = ' '.join([str(key) + '=' + str(val) for key, val in config_data.items()])
        args = {'-nvargs': dict_args}

    if isinstance(config_data, list):
        list_args = ' '.join(config_data)
        args = {'-args': list_args}

    m_type, m_dim, intercept = get_config(config_path)

    # If folder with Success files exists do not redo
    time = exec_func(exec_type, file_name, args, config_path)
    # Write Success File

    current_metrics = [algo, type, intercept, m_type, m_dim, str(time)]
    logging.info(','.join(current_metrics))


def perf_test_entry(family, algo, exec_type, mat_type, mat_shape, temp_dir, mode):

    algo_payload = {}

    if family is not None and algo is not None:
        for current_algo in algo:
            if current_algo == HAS_DUAL:
                algo_payload[current_algo] = family[0]
                if len(family) > 1:
                    print('While using {} we cannot have multiple families'.format(HAS_DUAL))
                    sys.exit()
            else:
                algo_payload[current_algo] = get_family(current_algo, ML_ALGO)

    elif algo is not None:
        for current_algo in algo:
            algo_payload[current_algo] = get_family(current_algo, ML_ALGO)

    elif family is not None:
        for current_family in family:
            algos = ML_ALGO[current_family]
            for current_algo in algos:
                algo_payload[current_algo] = current_family

    if 'data-gen' in mode:
        data_gen_dir = join(temp_dir, 'data-gen')
        create_dir(data_gen_dir)

        conf_packet = config_packets_datagen(algo_payload, mat_type, mat_shape, data_gen_dir)
        for family_name, config_files in conf_packet.items():
            for config in config_files:
                algorithm_workflow(family_name, exec_type, config,
                                   ML_GENDATA[family_name], 'data-gen')

    if 'train' in mode:
        data_gen_dir = join(temp_dir, 'data-gen')
        train_dir = join(temp_dir, 'train')
        create_dir(train_dir)

        conf_packet = config_packets_train(algo_payload, data_gen_dir, train_dir)
        for algo_name, config_files in conf_packet.items():
            for config in config_files:
                algorithm_workflow(algo_name, exec_type, config, algo_name, 'train')

    if 'predict' in mode:
        data_gen_dir = join(temp_dir, 'data-gen')
        train_dir = join(temp_dir, 'train')
        predict_dir = join(temp_dir, 'predict')
        create_dir(predict_dir)

        conf_packet = config_packets_predict(algo_payload, data_gen_dir, train_dir, predict_dir)
        for algo_name, config_files in conf_packet.items():
            for config in config_files:
                file_name = ML_PREDICT[algo_name]
                algorithm_workflow(algo_name, exec_type, config, file_name, 'predict')


if __name__ == '__main__':
    systemml_home = os.environ.get('SYSTEMML_HOME')

    # Default Arguments
    default_mat_type = ['dense', 'sparse']
    default_workload = ['data-gen', 'train', 'predict']
    default_mat_shape = ['10k_1k']
    default_temp_dir = join(systemml_home, 'scripts', 'perftest', 'temp')

    # Initialize time
    start_time = time.time()
    algo_flat = reduce(lambda x, y: x + y, ML_ALGO.values())

    # Argparse Module
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')
    cparser.add_argument('--family', help='specify class of algorithms (e.g regression, binomial)',
                         metavar='', choices=ML_ALGO.keys(), nargs='+')
    cparser.add_argument('--algo', help='specify the type of algorithm to run', metavar='',
                         choices=algo_flat, nargs='+')

    cparser.add_argument('-exec-type', default='singlenode', help='System-ML backend '
                         '(e.g singlenode, spark, spark-hybrid)', metavar='',
                         choices=['hybrid_spark', 'singlenode'])
    cparser.add_argument('--mat-type', default=default_mat_type, help='Type of matrix to generate '
                         '(e.g dense or sparse)', metavar='', choices=default_mat_type,
                         nargs='+')
    cparser.add_argument('--mat-shape', default=default_mat_shape, help='Shape of matrix '
                         'to generate (e.g 10k_1k)', metavar='', nargs='+')
    cparser.add_argument('-temp-dir', default=default_temp_dir, help='specify temporary directory',
                         metavar='')
    cparser.add_argument('--filename', default='perf_test', help='specify output file',
                         metavar='')
    cparser.add_argument('--mode', default=default_workload,
                         help='specify type of workload to run (e.g data-gen, train, predict)',
                         metavar='', choices=default_workload, nargs='+')

    args = cparser.parse_args()
    arg_dict = vars(args)

    # Debug arguments
    # print(arg_dict)

    # Check for validity of input arguments
    if args.family is not None:
        for fam in args.family:
            if fam not in ML_ALGO.keys():
                print('{} family not present in the performance test suit'.format(fam))
                sys.exit()

    if args.algo is not None:
        for algo in args.algo:
            if algo not in algo_flat:
                print('{} algorithm not present in the performance test suit'.format(args.algo))
                sys.exit()

    # Check dual condition
    if args.family is None:
        if HAS_DUAL in args.algo:
            print('cannot run {} without family type'.format(HAS_DUAL))
            sys.exit()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # Set level to 0 ->  debug
    # Set level to 20 -> metrics
    log_filename = args.filename + '_' + args.exec_type
    logging.basicConfig(filename=join(default_temp_dir, log_filename), level=0)
    logging.info('New performance test')
    logging.info('algorithm, run_type, intercept, matrix_type, data_shape, time_sec')

    # Remove filename item from dictionary
    del arg_dict['filename']
    perf_test_entry(**arg_dict)

    total_time = (time.time() - start_time)
    logging.info('Performance tests complete {0:.3f} secs \n'.format(total_time))
