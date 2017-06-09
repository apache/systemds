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

import argparse
import os

if __name__ == '__main__':
    cparser = argparse.ArgumentParser(description='SystemML Performance Test Script')
    cparser.add_argument('--class', help='specify class of algorithms (e.g regression, binomial, all)', metavar='')
    cparser.add_argument('--algo', help='specify the type of algorithm to run', metavar='')
    cparser.add_argument('-exec', default='singlenode', help='System-ML backend (e.g singlenode, spark, spark-hybrid)',
                         metavar='')
    cparser.add_argument('--matrix-type', help='Type of matrix to generate (e.g dense or sparse)',
                         metavar='')
    cparser.add_argument('--matrix-shape', help='Shape of matrix to generate (e.g dense or sparse)',
                         metavar='')

    # Optional Arguments
    cparser.add_argument('-temp-dir', help='specify temporary directory', metavar='')
    cparser.add_argument('--init', help='generate configuration files', metavar='')
    cparser.add_argument('--generate-data', help='generate data', metavar='')
    cparser.add_argument('--train', help='train algorithms', metavar='')
    cparser.add_argument('--predict', help='predict (if available)', metavar='')

    args = cparser.parse_args()
    arg_dict = vars(args)

