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

import pandas as pd
import argparse
import gspread
import pprint
from oauth2client.service_account import ServiceAccountCredentials
from functools import reduce


def auth(path, sheet_name):
    scope = ['https://spreadsheets.google.com/feeds']
    creds = ServiceAccountCredentials.from_json_keyfile_name(path, scope)
    gc = gspread.authorize(creds)
    sheet = gc.open("Perf").worksheet(sheet_name)
    return sheet


def get_data(sheet, tag):
    time = sheet.find('time_{}'.format(tag))
    algo = sheet.find('algo_{}'.format(tag))

    time_col = sheet.col_values(time.col)
    time_col = list(filter(lambda x: len(x) > 0, time_col))

    algo_col = sheet.col_values(algo.col)
    algo_col = list(filter(lambda x: len(x) > 0, algo_col))
    return algo_col, time_col


def get_data_dict(data_col):
    data_dict = {}
    all_algo = []
    for algo, _ in data_col:
        all_algo.append(algo)

    flatten_algo = reduce(lambda x, y: x+y, all_algo)
    filter_data = list(filter(lambda x: not x.startswith('algo_'), flatten_algo))
    distict_algos = set(filter_data)

    for algo_dist in distict_algos:
        for algo, time in data_col:
            for k, v in zip(algo, time):
                if algo_dist == k:
                    if algo_dist not in data_dict:
                        data_dict[k] = [v]
                    else:
                        data_dict[k].append(v)
    return data_dict

# ./stats.py --auth client_json.json --backend singlenode --tags 1.0 2.0
if __name__ == '__main__':
    execution_mode = ['hybrid_spark', 'singlenode']

    cparser = argparse.ArgumentParser(description='System-ML Statistics Script')
    cparser.add_argument('--auth', help='Location to read auth file',
                         required=True, metavar='')
    cparser.add_argument('--exec-mode', help='Execution mode', choices=execution_mode,
                         required=True, metavar='')
    cparser.add_argument('--tags', help='Tagging header value',
                         required=True, nargs='+')

    args = cparser.parse_args()
    arg_dict = vars(args)
    sheet = auth(args.auth, args.backend)
    all_data = sheet.get_all_records()

    data_col = []
    for tag in args.tags:
        algo_col, time_col = get_data(sheet, tag)
        data_col.append((algo_col, time_col))

    data_dict = get_data_dict(data_col)

    delta_algo = {}
    for k, v in data_dict.items():
        delta = float(v[0]) - float(v[1])
        delta_algo[k] = delta

    pprint.pprint(delta_algo, width=1)
