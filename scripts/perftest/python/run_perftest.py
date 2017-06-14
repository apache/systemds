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
import argparse
from functools import reduce
import os
from os.path import join
from utils import get_algo, config_reader
from configuration import init_conf, gen_data_config
import logging
import time
import json
import subprocess
from subprocess import Popen, PIPE, STDOUT
import itertools


ml_algo = {'binomial': ['MultiLogReg', 'l2-svm', 'm-svm'],
           'clustering': ['Kmeans'],
           'multinomial': ['naive-bayes', 'MultiLogReg', 'm-svm'],
           'regression': ['LinearRegDS', 'LinearRegCG', 'GLM'],
           'stats': ['Univar-Stats', 'bivar-stats', 'stratstats']}

datagen_dict = {'Kmeans': 'genRandData4Kmeans.dml'}

def exec_func(exec_type, algorithm, conf_path):

    if exec_type == 'singlenode':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-standalone.py')
    if exec_type == 'hybrid_spark':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-spark-submit.py')

    conf_dict = config_reader(conf_path)

    arg = []
    for key, val in conf_dict.items():
        if key == 'X':
            val = join(conf_path, val)
        if key == 'Y':
            val = join(conf_path, val)
        if key == 'YbyC':
            val = join(conf_path, val)
        arg.append('{}={}'.format(key, val))

    args = ' '.join(arg)
    print(algorithm)
    sys.exit()
    cmd = [exec_script, datagen_dict[algorithm], '-nvargs', args]
    cmd_string = ' '.join(cmd)


    return None


def perf_test_entry(family, algo, exec_type, mat_type, mat_shape, temp_dir, filename, mode):
    if algo is None:
        algo = get_algo(family, ml_algo)

    if 'data-gen' in mode:
        gen_config = gen_data_config(algo, mat_type, mat_shape, temp_dir)
        #for conf in gen_config:
        #    metrics = exec_func(exec_type, algo, conf)

        pass

    if 'train' in mode:
        # Create train dir
        # Create ini Files
        # train algo based on data generated
        # return metrics
        pass

    if 'predict' in mode:
        # Create predict dir
        # Create ini Files
        # train algo based on train data generated
        # return metrics
        pass

    report_path = join(temp_dir, filename)
    with open(report_path, "a") as pertest_report:
        pertest_report.write("appended text")

    return None


if __name__ == '__main__':
    systemml_home = os.environ.get('SYSTEMML_HOME')

    # Default Arguments
    default_mat_type = ['dense', 'sparse']
    default_workload = ['data-gen', 'train', 'predict']
    default_mat_shape = ['10k_1k']
    default_temp_dir = join(systemml_home, 'scripts', 'perftest', 'temp')

    # Initialize Logging
    start_time = time.time()
    logging.basicConfig(filename=join(default_temp_dir, 'perf_report.out'), level=logging.INFO)
    logging.info('New performance test')

    algo_flat = reduce(lambda x, y: x + y, ml_algo.values())

    # Argparse Module
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')
    group = cparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--family', help='specify class of algorithms (e.g regression, binomial)',
                       metavar='', choices=ml_algo.keys(), nargs='+')
    group.add_argument('--algo', help='specify the type of algorithm to run', metavar='',
                       choices=algo_flat, nargs='+')

    cparser.add_argument('-exec-type', default='singlenode', help='System-ML backend (e.g singlenode, '
                         'spark, spark-hybrid)', metavar='', choices=['hybrid_spark', 'singlenode'])
    cparser.add_argument('--mat-type', default=default_mat_type, help='Type of matrix to generate '
                         '(e.g dense or sparse)', metavar='', choices=default_mat_type,  nargs='+')
    cparser.add_argument('--mat-shape', default=default_mat_shape, help='Shape of matrix to generate '
                         '(e.g 10k_1k)', metavar='', nargs='+')

    cparser.add_argument('-temp-dir', default=default_temp_dir, help='specify temporary directory',
                         metavar='')
    cparser.add_argument('--filename', default='pertest.out', help='specify output file',
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
            if fam not in ml_algo.keys():
                print('{} family not present in the performance test suit'.format(fam))
                sys.exit()

    if args.algo is not None:
        for algo in args.algo:
            if algo not in algo_flat:
                print('{} algorithm not present in the performance test suit'.format(args.algo))
                sys.exit()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    perf_test_entry(**arg_dict)
    sys.exit()

    total_time = (time.time() - start_time)
    logging.info('Performance tests complete {0:.3f} secs \n'.format(total_time))
