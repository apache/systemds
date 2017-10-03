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
import subprocess
import shlex
import re
import tempfile
import os

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
    log_file = None
    is_temp_file = False

    if log_file_path is not None:
        log_file_path = log_file_path + '.log'
        log_file = open(log_file_path, "w+")
    else:
        os_log_file, log_file_path = tempfile.mkstemp()
        log_file = os.fdopen(os_log_file, 'w+')
        is_temp_file = True

    log_file.write(' '.join(exec_command))
    log_file.write('\n')
    proc1 = subprocess.Popen(exec_command, stdout=log_file,
                             stderr=subprocess.STDOUT)
    proc1.wait()
    return_code = proc1.returncode

    log_file.close()
    log_file = open(log_file_path, 'r+')

    if return_code == 0:
        if extract == 'time':
            return_data = parse_time(log_file)
        if extract == 'dir':
            return_data = parse_hdfs_paths(log_file)
        if extract == 'hdfs_base':
            return_data = parse_hdfs_base(log_file)
        if extract is None:
            return_data = 0

    if return_code != 0:
        return_data = 'proc_fail'
        print('sub-process failed, return code {}'.format(return_code))

    if is_temp_file:
        os.remove(log_file_path)

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
