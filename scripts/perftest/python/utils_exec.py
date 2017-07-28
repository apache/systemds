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

import subprocess
import shlex
import re


def get_std_out(process):
    out_arr = []
    while True:
        nextline = process.stdout.readline().decode('utf8').strip()
        out_arr.append(nextline)
        if nextline == '' and process.poll() is not None:
            break

    error_arr = []
    while True:
        nextline = process.stderr.readline().decode('utf8').strip()
        error_arr.append(nextline)
        if nextline == '' and process.poll() is not None:
            break

    return out_arr, error_arr


def subprocess_exec(cmd_string, extract=None):
    # Debug
    # print(cmd_string)
    proc1 = subprocess.Popen(shlex.split(cmd_string), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    error_arr, out_arr = get_std_out(proc1)
    std_outs = out_arr + error_arr
    return_data = proc1.returncode

    if extract == 'Time':
        return_data = parse_time(std_outs)
    if extract == 'dir':
        return_data = parse_dir(std_outs)

    return return_data


def parse_dir(std_outs):
    hdfs_dir = []
    for i in std_outs:
        if 'No such file or directory' in i:
            break
        elif 'hdfs' in i:
            current_dir = i.split(' ')[-1]
            hdfs_dir.append(current_dir)

    return hdfs_dir


def parse_time(raw_logs):
    """
    Parses raw input list and extracts time

    raw_logs : List
    Each line obtained from the standard output is in the list

    return: String
    Extracted time in seconds or time_not_found
    """
    # Debug
    # print(raw_logs)

    for line in raw_logs:
        if 'ERROR' in line:
            return 'error'
        if line.startswith('Total execution time'):
            extract_time = re.findall(r'\d+', line)
            total_time = '.'.join(extract_time)
            return total_time
    return 'time_not_found'
