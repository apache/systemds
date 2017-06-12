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

ml_algo = {'binomial': ['MultiLogReg', 'l2-svm', 'm-svm'],
           'clustering': ['Kmeans'],
           'multinomial': ['naive-bayes', 'MultiLogReg', 'm-svm'],
           'regression': ['LinearRegDS', 'LinearRegCG', 'GLM'],
           'stats': ['Univar-Stats', 'bivar-stats', 'stratstats']}


def main(family, algo, exec_type, mat_type, mat_shape, temp_dir, generate_data, train, predict):
    if algo is None:
        algo = get_algo(family, ml_algo)

    job = list(map(lambda x: int(x), [generate_data, train, predict]))
    init_conf(algo, temp_dir, mat_type, mat_shape, job)

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

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    main(**arg_dict)
