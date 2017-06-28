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

# TODO:

# Handle error both family and algo is specified (Invalid)
# If predict defined for clustering

# Smarter way to handle train algo file name
# Standardise variable constants with quotes

import sys
import time
import argparse
from functools import reduce
import os
from os.path import join
from utils import get_families, config_reader, create_dir, \
    exec_dml_and_parse_time, get_config, exec_test_data, check_predict
import logging
from datagen import config_packets_datagen
from train import config_packets_train
from predict import config_packets_predict

# A packet is a dictionary
# with key as the algorithm
# value as the list with configuration json files

# TODO:
# Detailed design
# Instructions to add a algo

ML_ALGO = {'binomial': ['MultiLogReg', 'l2-svm', 'm-svm'],
           'clustering': ['Kmeans'],
           'multinomial': ['naive-bayes', 'MultiLogReg', 'm-svm'],
           'regression1': ['LinearRegDS', 'LinearRegCG'],
           'regression2': ['GLM_poisson', 'GLM_gamma', 'GLM_binomial'],
           'stats1': ['Univar-Stats', 'bivar-stats'],
           'stats2': ['stratstats']}

ML_GENDATA = {'binomial': 'genRandData4LogisticRegression',
              'clustering': 'genRandData4Kmeans',
              'multinomial': 'genRandData4Multinomial',
              'regression1': 'genRandData4LogisticRegression',
              'regression2': 'genRandData4LogisticRegression',
              'stats1': 'genRandData4DescriptiveStats',
              'stats2': 'genRandData4StratStats'}

ML_TRAIN = {'GLM_poisson': 'GLM',
            'GLM_gamma': 'GLM',
            'GLM_binomial': 'GLM',
            'LinearRegCG': 'LinearRegCG',
            'LinearRegDS': 'LinearRegDS',
            'stratstats': 'stratstats',
            'Univar-Stats': 'Univar-Stats',
            'bivar-stats': 'bivar-stats',
            'Kmeans': 'Kmeans',
            'm-svm': 'm-svm',
            'l2-svm': 'l2-svm',
            'MultiLogReg': 'MultiLogReg',
            'naive-bayes': 'naive-bayes'}

ML_PREDICT = {'Kmeans': 'Kmeans-predict',
              'LinearRegCG': 'GLM-predict',
              'LinearRegDS': 'GLM-predict',
              'm-svm': 'm-svm-predict',
              'l2-svm': 'l2-svm-predict',
              'MultiLogReg': 'GLM-predict',
              'naive-bayes': 'naive-bayes-predict'}


EXCLUDE_TEST_SPLIT = ['stats1', 'stats2']


# Responsible for execution and metric logging
def algorithm_workflow(algo, exec_type, config_path, file_name, action_mode):
    """
    This function is responsible for overall workflow. This does the following actions
    Check if the input is key value argument or list of positional args
    Execution and time
    Logging Metrics


    algo : String
    exec_type : String
    config_path : String
    file_name : String
    action_mode : String
    """

    config_data = config_reader(config_path + '.json')

    if isinstance(config_data, dict):
        dict_args = ' '.join([str(key) + '=' + str(val) for key, val in config_data.items()])
        args = {'-nvargs': dict_args}

    if isinstance(config_data, list):
        list_args = ' '.join(config_data)
        args = {'-args': list_args}

    #m_type, m_dim, intercept = get_config(config_path)
    #current_metrics = [algo, action_mode, intercept, m_type, m_dim, str(time)]

    last_name = config_path.split('/')[-1]

    time = exec_dml_and_parse_time(exec_type, file_name, args, config_path)
    current_metrics = [algo, action_mode, exec_type, time, last_name]

    print('{},{},{} '.format(algo, action_mode, time))

    logging.info(','.join(current_metrics))


# Perf test entry point
def perf_test_entry(family, algo, exec_type, mat_type, mat_shape, temp_dir, mode):
    """
    This function is the entry point for the algorithms

    """

    # algos to run is a list of tuples with
    # [(m-svm, binomial), (m-svm, multinomial)...]
    # Basic block for execution of scripts
    algos_to_run = []

    # Sections below build algos_to_run in our performance test
    # Handles algorithms like m-svm and MultiLogReg which have multiple
    # data generation scripts (dual datagen)
    # --family is taken into consideration only when there are multiple datagen for an algo

    if family is not None and algo is not None:
        for current_algo in algo:
            family_list = get_families(current_algo, ML_ALGO)
            if len(family_list) == 1:
                algos_to_run.append((current_algo, family_list[0]))
            else:
                intersection = set(family).intersection(family_list)
                for valid_family in intersection:
                    algos_to_run.append((current_algo, valid_family))

    # When the user inputs just algorithms to run
    elif algo is not None:
        for current_algo in algo:
            family_list = get_families(current_algo, ML_ALGO)
            for f in family_list:
                algos_to_run.append((current_algo, f))

    # When the user just specifies only families to run
    elif family is not None:
        for current_family in family:
            algos = ML_ALGO[current_family]
            for current_algo in algos:
                algos_to_run.append((current_algo, current_family))

    if 'data-gen' in mode:
        data_gen_dir = join(temp_dir, 'data-gen')
        create_dir(data_gen_dir)
        conf_packet = config_packets_datagen(algos_to_run, mat_type, mat_shape, data_gen_dir)
        for family_name, config_folders in conf_packet.items():
            for config in config_folders:
                file_name = ML_GENDATA[family_name]
                algorithm_workflow(family_name, exec_type, config, file_name, 'data-gen')
                if family_name not in EXCLUDE_TEST_SPLIT:
                    exec_test_data(exec_type, config)

    if 'train' in mode:
        data_gen_dir = join(temp_dir, 'data-gen')
        train_dir = join(temp_dir, 'train')
        create_dir(train_dir)

        conf_packet = config_packets_train(algos_to_run, data_gen_dir, train_dir)
        for algo_name, config_files in conf_packet.items():
            for config in config_files:
                file_name = ML_TRAIN[algo_name]
                algorithm_workflow(algo_name, exec_type, config, file_name, 'train')

    if 'predict' in mode:
        data_gen_dir = join(temp_dir, 'data-gen')
        train_dir = join(temp_dir, 'train')
        predict_dir = join(temp_dir, 'predict')
        create_dir(predict_dir)
        algos_to_run_perdict = list(filter(lambda algo: check_predict(algo[0], ML_PREDICT), algos_to_run))
        if len(algos_to_run_perdict) < 0:
            pass

        conf_packet = config_packets_predict(algos_to_run_perdict, data_gen_dir, train_dir, predict_dir)

        for algo_name, config_files in conf_packet.items():
                for config in config_files:
                    file_name = ML_PREDICT[algo_name]
                    algorithm_workflow(algo_name, exec_type, config, file_name, 'predict')

if __name__ == '__main__':

    # sys ml env set and error handling
    systemml_home = os.environ.get('SYSTEMML_HOME')
    if systemml_home is None:
        print('SYSTEMML_HOME not found')
        sys.exit()

    # Default Arguments
    default_mat_type = ['dense', 'sparse']
    default_workload = ['data-gen', 'train', 'predict']
    default_mat_shape = ['10k_100']

    # Default temp directory, contains everything generated in perftest
    default_temp_dir = join(systemml_home, 'scripts', 'perftest', 'temp')
    create_dir(default_temp_dir)

    # Initialize time
    start_time = time.time()

    # Remove duplicates algorithms and used as default inputs
    all_algos = set(reduce(lambda x, y: x + y, ML_ALGO.values()))

    # Argparse Module
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')
    cparser.add_argument('--family', help='specify class of algorithms (e.g regression, binomial)',
                         metavar='', choices=ML_ALGO.keys(), nargs='+')
    cparser.add_argument('--algo', help='specify the type of algorithm to run (Overrides --family)', metavar='',
                         choices=all_algos, nargs='+')

    cparser.add_argument('--exec-type', default='singlenode', help='System-ML backend '
                         '(e.g singlenode, spark, spark-hybrid)', metavar='',
                         choices=['hybrid_spark', 'singlenode'])
    cparser.add_argument('--mat-type', default=default_mat_type, help='Type of matrix to generate '
                         '(e.g dense or sparse)', metavar='', choices=default_mat_type,
                         nargs='+')
    cparser.add_argument('--mat-shape', default=default_mat_shape, help='Shape of matrix '
                         'to generate (e.g 10k_1k)', metavar='', nargs='+')
    cparser.add_argument('--temp-dir', default=default_temp_dir, help='specify temporary directory',
                         metavar='')
    cparser.add_argument('--filename', default='perf_test', help='specify output file for the perf'
                         ' metics', metavar='')
    cparser.add_argument('--mode', default=default_workload,
                         help='specify type of workload to run (e.g data-gen, train, predict)',
                         metavar='', choices=default_workload, nargs='+')

    # Args is a namespace
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
            if algo not in all_algos:
                print('{} algorithm not present in the performance test suit'.format(args.algo))
                sys.exit()

        # This section check the validity of dual datagen algorithms like m-svm
        algo_families = {}
        for current_algo in args.algo:
            algo_families[current_algo] = get_families(current_algo, ML_ALGO)

        if len(algo_families[current_algo]) > 1:
            if args.family is None:
                print('family should be present for {}'.format(current_algo))
                sys.exit()

            valid_families = set(algo_families[current_algo])
            input_families = set(args.family)
            common_families = input_families.intersection(valid_families)
            if len(common_families) == 0:
                print('Please specify a valid family for {} and the '
                      'valid families are {}'.format(current_algo, ' '.join(valid_families)))
                sys.exit()

    # Set level to 0 -> debug mode
    # Set level to 20 -> Plain metrics
    log_filename = args.filename + '_' + args.exec_type + '.out'
    logging.basicConfig(filename=join(default_temp_dir, log_filename), level=20)
    logging.info('New performance test')
    logging.info('algorithm, run_type, intercept, matrix_type, data_shape, time_sec')

    # Remove filename item from dictionary
    del arg_dict['filename']

    perf_test_entry(**arg_dict)

    total_time = (time.time() - start_time)
    logging.info('Performance tests complete {0:.3f} secs \n'.format(total_time))
