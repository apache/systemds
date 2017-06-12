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
# Handel Intercept

import sys
import argparse
from functools import reduce
import os
from os.path import join
from utils import get_algo
from configuration import init_conf
import logging
import time
import json
import subprocess
from subprocess import Popen, PIPE, STDOUT

ml_algo = {'binomial': ['MultiLogReg', 'l2-svm', 'm-svm'],
           'clustering': ['Kmeans'],
           'multinomial': ['naive-bayes', 'MultiLogReg', 'm-svm'],
           'regression': ['LinearRegDS', 'LinearRegCG', 'GLM'],
           'stats': ['Univar-Stats', 'bivar-stats', 'stratstats']}

datagen_dict = {'kmeans': 'genRandData4Kmeans.dml'}


def exec_conf(exec_type, temp_dir, mat_shape):
    # train
    # predict

    conf_dir = join(temp_dir, 'conf')
    data_gen_dir = join(temp_dir, 'data_gen')

    config_files = os.listdir(conf_dir)
    config_data_gen = list(filter(lambda x: 'datagen' in x, config_files))
    config_train = list(filter(lambda x: 'train' in x, config_files))
    config_predict = list(filter(lambda x: 'predict' in x, config_files))

    if exec_type == 'singlenode':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-standalone.py')
    if exec_type == 'hybrid_spark':
        exec_script = join(os.environ.get('SYSTEMML_HOME'), 'bin', 'systemml-spark-submit.py')

    for data_gen in config_data_gen:
        mat_type = data_gen.split('_')[0]
        gen_algo = data_gen.split('_')[1]
        gen_dirname = data_gen.split('.')[0]
        shape_index = int(data_gen.split('_')[3][0])

        exists_dir = join(data_gen_dir, gen_dirname)
        if os.path.exists(exists_dir):
            print('{} already exist continue...'.format(gen_dirname))
            continue

        file_path = join(conf_dir, data_gen)

        with open(file_path, 'r') as f:
            current_config = json.load(f)

        arg = []
        for key, val in current_config.items():
            if key == 'X':
                val = join(data_gen_dir, gen_dirname, val)
            if key == 'Y':
                val = join(data_gen_dir, gen_dirname, val)
            if key == 'YbyC':
                val = join(data_gen_dir, gen_dirname, val)
            arg.append('{}={}'.format(key, val))

        args = ' '.join(arg)
        cmd = [exec_script, datagen_dict[gen_algo], '-nvargs', args]
        cmd_string = ' '.join(cmd)
        run_time = time.time()
        return_code = subprocess.call(cmd_string, shell=True)
        total_time = time.time() - run_time

        log_info = [gen_algo, 'data_gen', mat_type, mat_shape[shape_index],
                    '{0:.3f}'.format(total_time), str(return_code)]
        logging.critical(','.join(log_info))

    pass


def perf_test_entry(family, algo, exec_type, mat_type, mat_shape, temp_dir, generate_data, train, predict):
    if algo is None:
        algo = get_algo(family, ml_algo)

    job = list(map(lambda x: int(x), [generate_data, train, predict]))
    init_conf(algo, temp_dir, mat_type, mat_shape, job)

    exec_conf(exec_type, temp_dir, mat_shape)

    return None


if __name__ == '__main__':
    algo_flat = reduce(lambda x, y: x + y, ml_algo.values())
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')
    group = cparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--family', help='specify class of algorithms (e.g regression, binomial)', metavar='',
                       choices=ml_algo.keys(), nargs='+')
    group.add_argument('--algo', help='specify the type of algorithm to run', metavar='',
                       choices=algo_flat, nargs='+')

    cparser.add_argument('-exec-type', default='singlenode', help='System-ML backend (e.g singlenode, '
                                                                  'spark, spark-hybrid)', metavar='',
                         choices=['hybrid_spark', 'singlenode'])
    cparser.add_argument('--mat-type', default='dense', help='Type of matrix to generate (e.g dense '
                                                             'or sparse)', metavar='', choices=['sparse', 'dense'])
    cparser.add_argument('--mat-shape', help='Shape of matrix to generate (e.g '
                                             '10k_1k)', metavar='', nargs='+')

    # Optional Arguments
    cparser.add_argument('-temp-dir', help='specify temporary directory', metavar='')
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
            if algo not in algo_flat:
                print('{} algorithm not present in the performance test suit'.format(args.algo))
                sys.exit()

    if args.temp_dir is None:
        systemml_home = os.environ.get('SYSTEMML_HOME')
        args.temp_dir = join(systemml_home, 'scripts', 'perftest', 'temp')

    start_time = time.time()
    logging.basicConfig(filename=join(args.temp_dir, 'perftest.out'), level=logging.INFO)
    logging.info('New experiment state time {}'.format(start_time))
    logging.info(args)
    log_header = ['algorithm', 'run_type', 'mat_typ', 'mat_shape', 'time', 'return_code']
    logging.critical(','.join(log_header))

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    perf_test_entry(**arg_dict)
