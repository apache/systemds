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

import sys
import subprocess
import shlex
import re

# Subprocess and log parsing related functions


def subprocess_exec(cmd_string, log_file_path=None, extract=None):
    """
    Execute the input string as subprocess

    cmd_string: String
    Input string to be executed as a sub process

    extract: String
    Based on extract as time/dir we extract this information from
    the logs accordingly

    log_file_path: String
    Path to write the log file

    return: String
    Based on extract we return the relevant string
    """
    # Debug
    # print(cmd_string)

    exec_command = shlex.split(cmd_string)
    proc1 = subprocess.Popen(exec_command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    error_arr, out_arr = get_all_logs(proc1)
    std_outs = out_arr + error_arr
    std_outs.insert(0, ' '.join(exec_command))
    return_code = proc1.returncode

    if log_file_path is not None:
        write_logs(std_outs, log_file_path + '.log')

    if return_code == 0:
        if extract == 'time':
            return_data = parse_time(std_outs)
        if extract == 'dir':
            return_data = parse_hdfs_paths(std_outs)
        if extract == 'hdfs_base':
            return_data = parse_hdfs_base(std_outs)
        if extract is None:
            return_data = 0

    if return_code != 0:
        return_data = 'proc_fail'
        print('sub-process failed, return code {}'.format(return_code))

    return return_data


def parse_hdfs_base(std_outs):
    """
    return: String
    hdfs base uri
    """

    hdfs_uri = None
    for line in std_outs:
        if line.startswith('hdfs://'):
            hdfs_uri = line
    if hdfs_uri is None:
        sys.exit('HDFS URI not found')
    return hdfs_uri


def write_logs(std_outs, log_file_path):
    """
    Write all logs to the specified location
    """

    with open(log_file_path, 'w')as log:
        log.write("\n".join(std_outs))


def get_all_logs(process):
    """
    Based on the subprocess capture logs

    process: Process
    Process object

    return: List, List
    Std out and Error as logs as list
    """

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


def parse_hdfs_paths(std_outs):
    """
    Extract the hdfs paths from the input

    std_outs: List
    Std outs obtained from the subprocess

    return: List
    Obtain a list of hdfs paths
    """

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
