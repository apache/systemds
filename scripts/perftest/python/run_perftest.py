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
import logging
from datetime import datetime
from datagen import config_packets_datagen
from train import config_packets_train
from predict import config_packets_predict
from utils import get_families, config_reader, \
    exec_dml_and_parse_time, exec_test_data, check_predict, get_folder_metrics, args_dict_split, \
    sup_args
from file_system import create_dir_local, write_success, get_existence

# A packet is a dictionary
# with key as the algorithm
# value as the list with configuration json files


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
              'naive-bayes': 'naive-bayes-predict',
              'GLM_poisson': 'GLM-predict',
              'GLM_gamma': 'GLM-predict',
              'GLM_binomial': 'GLM-predict'}

DENSE_TYPE_ALGOS = ['clustering', 'stats1', 'stats2']

sup_args_dict = {}


# Responsible for execution and metric logging
def algorithm_workflow(algo, exec_type, config_path, dml_file_name, action_mode, current_dir):
    """
    This function is responsible for overall workflow. This does the following actions
    Check if the input is key value argument or list of positional args
    Execution and time
    Logging Metrics

    algo : String
    Input algorithm specified

    exec_type : String
    Contains the execution type singlenode / hybrid_spark

    config_path : String
    Path to read the json file from

    dml_file_name : String
    DML file name to be used while processing the arguments give

    action_mode : String
    Type of action data-gen, train ...
    """
    config_data = config_reader(config_path + '.json')

    if isinstance(config_data, dict):
        dict_args = ' '.join([str(key) + '=' + str(val) for key, val in config_data.items()])
        args = {'-nvargs': dict_args}

    if isinstance(config_data, list):
        list_args = ' '.join(config_data)
        args = {'-args': list_args}

    config_file_name = config_path.split('/')[-1]
    mat_type, mat_shape, intercept = get_folder_metrics(config_file_name, action_mode)

    temp_cwd = join(current_dir, config_file_name)

    # temp_dir_exist
    exit_flag_success = get_existence(temp_cwd)

    if exit_flag_success:
        time = 'data_exists'
    else:
        time = exec_dml_and_parse_time(exec_type, dml_file_name, args, spark_args_dict, sup_args_dict)
        write_success(time, temp_cwd)

    print('{},{},{},{},{},{}'.format(algo, action_mode, intercept, mat_type, mat_shape, time))
    current_metrics = [algo, action_mode, intercept, mat_type, mat_shape, time]
    logging.info(','.join(current_metrics))


def perf_test_entry(family, algo, exec_type, mat_type, mat_shape, config_dir, mode, temp_dir):
    """
    This function is the entry point for performance testing

    family: List
    A family may contain one or more algorithm based on data generation script used

    algo: List
    Input algorithms

    exec_type: String
    Contains the execution type singlenode / hybrid_spark

    mat_type: List
    Type of matrix to generate dense, sparse, all

    mat_shape: List
    Dimensions of the input matrix with rows and columns

    config_dir: String
    Location to store all configuration

    mode: List
    Type of workload to run. data-gen, train ...

    temp_dir: String
    Location to store all output files created during perf test
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
        # Create config directories
        data_gen_config_dir = join(config_dir, 'data-gen')
        create_dir_local(data_gen_config_dir)

        # Create output path
        data_gen_dir = join(temp_dir, 'data-gen')
        conf_packet = config_packets_datagen(algos_to_run, mat_type, mat_shape, data_gen_dir,
                                             DENSE_TYPE_ALGOS, data_gen_config_dir)

        for family_name, config_folders in conf_packet.items():
            for config in config_folders:
                file_name = ML_GENDATA[family_name]
                algorithm_workflow(family_name, exec_type, config, file_name, 'data-gen', data_gen_dir)
                # Statistic family do not require to be split
                if family_name not in ['stats1', 'stats2']:
                    if data_gen_dir.startswith('hdfs'):
                        exec_test_data(exec_type, spark_args_dict, sup_args_dict, config)
                    else:
                        exec_test_data(exec_type, spark_args_dict, sup_args_dict, config)

    if 'train' in mode:
        # Create config directories
        train_config_dir = join(config_dir, 'train')
        create_dir_local(train_config_dir)

        # Create output path
        data_gen_dir = join(temp_dir, 'data-gen')
        train_dir = join(temp_dir, 'train')

        conf_packet = config_packets_train(algos_to_run, mat_type, mat_shape, data_gen_dir,
                                           train_dir, DENSE_TYPE_ALGOS, train_config_dir)
        for algo_name, config_files in conf_packet.items():
            for config in config_files:
                file_name = ML_TRAIN[algo_name]
                algorithm_workflow(algo_name, exec_type, config, file_name, 'train', train_dir)

    if 'predict' in mode:
        # Create config directories
        predict_config_dir = join(config_dir, 'predict')
        create_dir_local(predict_config_dir)

        # Create output path
        data_gen_dir = join(temp_dir, 'data-gen')
        train_dir = join(temp_dir, 'train')
        predict_dir = join(temp_dir, 'predict')

        algos_to_run = list(filter(lambda algo: check_predict(algo[0], ML_PREDICT), algos_to_run))
        if len(algos_to_run) < 1:
            # No algorithms with predict found
            pass
        conf_packet = config_packets_predict(algos_to_run, mat_type, mat_shape, data_gen_dir,
                                             train_dir, predict_dir, DENSE_TYPE_ALGOS,
                                             predict_config_dir)
        for algo_name, config_files in conf_packet.items():
                for config in config_files:
                    file_name = ML_PREDICT[algo_name]
                    algorithm_workflow(algo_name, exec_type, config, file_name, 'predict', predict_dir)


if __name__ == '__main__':
    # sys ml env set and error handling
    systemml_home = os.environ.get('SYSTEMML_HOME')
    if systemml_home is None:
        print('SYSTEMML_HOME not found')
        sys.exit()

    # Supported Arguments
    mat_type = ['dense', 'sparse', 'all']
    workload = ['data-gen', 'train', 'predict']
    execution_mode = ['hybrid_spark', 'singlenode']
    # Default Arguments
    default_mat_shape = ['10k_100']

    # Default temp directory, contains everything generated in perftest
    default_temp_dir = join(systemml_home, 'scripts', 'perftest', 'temp')

    # Initialize time
    start_time = time.time()

    # Default Date Time
    time_now = str(datetime.now())

    # Remove duplicates algorithms and used as default inputs
    all_algos = set(reduce(lambda x, y: x + y, ML_ALGO.values()))

    # Families
    all_families = ML_ALGO.keys()

    # Argparse Module
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')
    cparser.add_argument('--family', help='space separated list of classes of algorithms '
                         '(available : ' + ', '.join(sorted(all_families)) + ')',
                         metavar='', choices=all_families, nargs='+')
    cparser.add_argument('--algo', help='space separated list of algorithm to run '
                         '(Overrides --family, available : ' + ', '.join(sorted(all_algos)) + ')', metavar='',
                         choices=all_algos, nargs='+')

    cparser.add_argument('--exec-type', default='hybrid_spark', help='System-ML backend '
                         'available : ' + ','.join(execution_mode), metavar='',
                         choices=execution_mode)
    cparser.add_argument('--mat-type', default=['all'], help='space separated list of types of matrix to generate '
                         'available : ' + ','.join(mat_type), metavar='', choices=mat_type,
                         nargs='+')
    cparser.add_argument('--mat-shape', default=default_mat_shape, help='space separated list of shapes of matrices '
                         'to generate (e.g 10k_1k, 20M_4k)', metavar='', nargs='+')

    cparser.add_argument('--config-dir', default=default_temp_dir, help='temporary directory '
                         'where generated, training and prediction data is put', metavar='')
    cparser.add_argument('--filename', default='perf_test', help='name of the output file for the perf'
                         ' metrics', metavar='')
    cparser.add_argument('--mode', default=workload,
                         help='space separated list of types of workloads to run (available: data-gen, train, predict)',
                         metavar='', choices=workload, nargs='+')
    # Change this to temp-dir
    cparser.add_argument('--temp-dir', default=default_temp_dir,
                         help='define the file system to work on', metavar='')

    # Configuration Options
    cparser.add_argument('-stats', help='Monitor and report caching/recompilation statistics, '
                                        'heavy hitter <count> is 10 unless overridden', nargs='?', const='10',
                         metavar='')
    cparser.add_argument('-explain', help='explains plan levels can be hops, runtime, '
                                          'recompile_hops, recompile_runtime', nargs='?', const='runtime', metavar='')
    cparser.add_argument('-config', help='System-ML configuration file (e.g SystemML-config.xml)', metavar='')

    # Spark Configuration Option
    cparser.add_argument('--master', help='local, yarn-client, yarn-cluster', metavar='')
    cparser.add_argument('--driver-memory', help='Memory for driver (e.g. 512M)', metavar='')
    cparser.add_argument('--num-executors', help='Number of executors to launch', metavar='')
    cparser.add_argument('--executor-memory', help='Memory per executor', metavar='')
    cparser.add_argument('--executor-cores', help='Number of cores', metavar='')
    cparser.add_argument('--conf', help='Spark configuration file', nargs='+', metavar='')

    # Args is a namespace
    args = cparser.parse_args()
    all_arg_dict = vars(args)
    arg_dict, config_dict, spark_dict = args_dict_split(all_arg_dict)

    create_dir_local(default_temp_dir)

    # Global variables
    sup_args_dict, spark_args_dict = sup_args(config_dict, spark_dict, args.exec_type)

    # Debug arguments
    # print(arg_dict)

    # default_mat_type validity
    if len(args.mat_type) > 2:
        print('length of --mat-type argument cannot be greater than two')
        sys.exit()

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
    logging.info('New performance test started at {}'.format(time_now))
    logging.info('algorithm,run_type,intercept,matrix_type,data_shape,time_sec')

    # Remove filename item from dictionary as its already used to create the log above
    del arg_dict['filename']
    perf_test_entry(**arg_dict)

    total_time = (time.time() - start_time)
    logging.info('Performance tests complete {0:.3f} secs \n'.format(total_time))
