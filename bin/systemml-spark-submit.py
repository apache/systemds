#!/usr/bin/env python
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

import os
import sys
from os.path import join, exists
from os import environ


# error help print
def print_usage_and_exit():
    this_script = sys.argv[0]
    print('Usage: ' + this_script + ' <dml-filename> [arguments]')
    sys.exit(1)

if environ.get('SPARK_HOME') is None:
    print('SPARK_HOME not set')

if len(sys.argv) < 2:
    print('Wrong usage')
    print_usage_and_exit()

# find the systemML root path which contains the bin folder, the script folder and the target folder
# tolerate path with spaces
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root_dir = os.path.dirname(script_dir)
user_dir = os.getcwd()

scripts_dir = join(project_root_dir, 'scripts')
build_dir = join(project_root_dir, 'target')
target_jars = build_dir + '/' + '*.jar'


build_err_msg = 'You must build the project before running this script.'
build_dir_err_msg = 'Could not find target directory ' + build_dir + '. ' + build_err_msg


if not(exists(build_dir)):
    print(build_dir_err_msg)
    sys.exit(1)

print('================================================================================')

# if the present working directory is the project root or bin folder, then use the temp folder as user.dir
if user_dir == project_root_dir or user_dir == join(project_root_dir, 'bin'):
    user_dir = join(project_root_dir, 'temp')
    print('Output dir: ' + user_dir)

# if the SystemML-config.xml does not exist, create it from the template
systemml_config_path = join(project_root_dir, 'conf', 'SystemML-config.xml')
systemml_template_config_path = join(project_root_dir, 'conf', 'SystemML-config.xml.template')
if not(exists(systemml_config_path)):
    shutil.copyfile(systemml_template_config_path, systemml_config_path)
    print('... created ' + systemml_config_path)


script_file = sys.argv[1]


# from http://stackoverflow.com/questions/1724693/find-a-file-in-python
def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return join(root, name)
    return None

# if the script file path was omitted, try to complete the script path
if not(exists(script_file)):
    script_file_name = os.path.abspath(script_file)
    script_file_found = find_file(script_file, scripts_dir)
    if script_file_found is None:
        print('Could not find DML script: ' + script_file)
        print_usage_and_exit()
    else:
        script_file = script_file_found
        print('DML Script:' + script_file)

cmd = ['$SPARK_HOME/bin/spark-submit', '--jars', target_jars, '--f', script_file,
       '-exec spark', '-config', systemml_config_path] + sys.argv[2:]

return_code = os.system(' '.join(cmd))
# For debugging
# print(' '.join(cmd))

return_code = os.system(' '.join(cmd))

if return_code != 0:
    print('Failed to run SystemML. Exit code :' + str(return_code))
    print(' '.join(cmd))
