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
import argparse


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

parser = argparse.ArgumentParser(description='System-ML Spark Submit Script', add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--help', action='help', help='Print this usage message and exit')
parser.add_argument('--master', default='local[*]', help='local, yarn-client, yarn-cluster', metavar='\b')
parser.add_argument('--driver-memory', default='5G', help='Memory for driver (e.g. 512M)', metavar='\b')
parser.add_argument('--num-executors', default='2', help='Number of executors to launch', metavar='\b')
parser.add_argument('--executor-memory', default='2G', help='Memory per executor', metavar='\b')
parser.add_argument('--executor-cores', default='1', help='Number of cores', metavar='\b')
parser.add_argument('--conf', default='', help='Configuration settings', metavar='\b')
parser.add_argument('-f', required=True, help='DML script file name', metavar='\b')
parser.add_argument('-nvargs', nargs='*')
args = parser.parse_args()
arg_dict = vars(args)

# find the systemML root path which contains the bin folder, the script folder and the target folder
# tolerate path with spaces
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root_dir = os.path.dirname(script_dir)
user_dir = os.getcwd()

scripts_dir = join(project_root_dir, 'scripts')
build_dir = join(project_root_dir, 'target')
target_jars = build_dir + '/' + '*.jar'
log4j_properties_path = join(project_root_dir, 'conf', 'log4j.properties')


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


script_file = arg_dict['f']

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

log_conf = '--conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:{}" '.format(log4j_properties_path)
max_result_conf = '--conf spark.driver.maxResultSize=0 '
frame_size_conf = '--conf spark.akka.frameSize=128 '
default_conf = log_conf + max_result_conf + frame_size_conf + arg_dict['conf']


cmd = ['$SPARK_HOME/bin/spark-submit', '--master', arg_dict['master'], '--driver-memory', arg_dict['driver_memory'],
       '--num-executors', arg_dict['num_executors'], '--executor-memory', arg_dict['executor_memory'],
       '--executor-cores', arg_dict['executor_cores'], default_conf, '--jars', target_jars, '-f', script_file,
       '-exec hybrid_spark', '-config', systemml_config_path, '-nvargs ' + ' '.join(arg_dict['nvargs'])]

return_code = os.system(' '.join(cmd))
# For debugging
# print(' '.join(cmd))

return_code = os.system(' '.join(cmd))

if return_code != 0:
    print('Failed to run SystemML. Exit code :' + str(return_code))
    print(' '.join(cmd))

