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

import argparse
import os
import pprint
from os.path import join
import matplotlib.pyplot as plt
from gdocs_utils import auth


# Dict
# {algo_name : [algo_1.0': t1, 'algo_2.0': t2]}
def get_formatted_data(sheet_data):
    """
    Read all the data from google sheets and transforms it into a dictionary that can be
    use for plotting later
    """
    algo_dict = {}

    for i in sheet_data:
        inn_count = 0
        data = []
        for key, val in i.items():
            inn_count += 1
            if inn_count < 3:
                data.append(key)
                data.append(val)

            if inn_count == 2:
                t1, v1, _, v2 = data
                if len(str(v2)) > 0:
                    if v1 not in algo_dict:
                        algo_dict[v1] = [{t1: v2}]
                    else:
                        algo_dict[v1].append({t1: v2})
                    inn_count = 0
                    data = []
    return algo_dict


def plot(x, y, xlab, ylab, title):
    """
    Save plots to the current folder based on the arguments
    """
    CWD = os.getcwd()
    PATH = join(CWD, title)
    width = .35
    plt.bar(x, y, color="red", width=width)
    plt.xticks(x)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(PATH + '.png')
    print('Plot {} generated'.format(title))
    return plt

# Example Usage
#  ./stats.py --auth ../key/client_json.json --exec-mode singlenode
if __name__ == '__main__':
    execution_mode = ['hybrid_spark', 'singlenode']

    cparser = argparse.ArgumentParser(description='System-ML Statistics Script')
    cparser.add_argument('--auth', help='Location to read auth file',
                         required=True, metavar='')
    cparser.add_argument('--exec-type', help='Execution mode', choices=execution_mode,
                         required=True, metavar='')
    cparser.add_argument('--plot', help='Algorithm to plot', metavar='')

    args = cparser.parse_args()

    sheet = auth(args.auth, args.exec_type)
    all_data = sheet.get_all_records()

    plot_data = get_formatted_data(all_data)
    if args.plot is not None:
        print(plot_data[args.plot])
        title = args.plot
        ylab = 'Time in sec'
        xlab = 'Version'
        x = []
        y = []
        for i in plot_data[args.plot]:
            version = list(i.keys())[0]
            time = list(i.values())[0]
            y.append(time)
            x.append(version)

        x = list(map(lambda x: float(x.split('_')[1]), x))
        plot(x, y, xlab, ylab, title)
    else:
        pprint.pprint(plot_data, width=1)