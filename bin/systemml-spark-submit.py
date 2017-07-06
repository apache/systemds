#!/usr/bin/env python
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

import os
import sys
from os.path import join, exists, abspath
from os import environ
import glob
import argparse
import shutil
import platform

if environ.get('SPARK_HOME') is None:
    print('SPARK_HOME not set')
    sys.exit(1)
else:
    spark_home = environ.get('SPARK_HOME')
    spark_path = join(spark_home, 'bin', 'spark-submit')


# error help print
def print_usage_and_exit():
    print('Usage: ./systemml-spark-submit.py -f <dml-filename> [arguments]')
    sys.exit(1)

cparser = argparse.ArgumentParser(description='System-ML Spark Submit Script')

# SPARK-SUBMIT Options
cparser.add_argument('--master', default='local[*]', help='local, yarn-client, yarn-cluster', metavar='')
cparser.add_argument('--driver-memory', default='5G', help='Memory for driver (e.g. 512M)', metavar='')
cparser.add_argument('--num-executors', default='2', help='Number of executors to launch', metavar='')
cparser.add_argument('--executor-memory', default='2G', help='Memory per executor', metavar='')
cparser.add_argument('--executor-cores', default='1', help='Number of cores', metavar='')
cparser.add_argument('--conf', help='Spark configuration file', nargs='+', metavar='')

# SYSTEM-ML Options
cparser.add_argument('-nvargs', help='List of attributeName-attributeValue pairs', nargs='+', metavar='')
cparser.add_argument('-args', help='List of positional argument values', metavar='', nargs='+')
cparser.add_argument('-config', help='System-ML configuration file (e.g SystemML-config.xml)', metavar='')
cparser.add_argument('-exec', default='hybrid_spark', help='System-ML backend (e.g spark, spark-hybrid)', metavar='')
cparser.add_argument('-explain', help='explains plan levels can be hops, runtime, '
                                      'recompile_hops, recompile_runtime', nargs='?', const='runtime', metavar='')
cparser.add_argument('-debug', help='runs in debug mode', action='store_true')
cparser.add_argument('-stats', help='Monitor and report caching/recompilation statistics, '
                                    'heavy hitter <count> is 10 unless overridden', nargs='?', const='10', metavar='')
cparser.add_argument('-gpu', help='uses CUDA instructions when reasonable, '
                                  'set <force> option to skip conservative memory estimates '
                                  'and use GPU wherever possible', nargs='?')
cparser.add_argument('-f', required=True, help='specifies dml/pydml file to execute; '
                                               'path can be local/hdfs/gpfs', metavar='')

args = cparser.parse_args()

# Optional arguments
ml_options = []
if args.nvargs is not None:
    ml_options.append('-nvargs')
    ml_options.append(' '.join(args.nvargs))
if args.args is not None:
    ml_options.append('-args')
    ml_options.append(' '.join(args.args))
if args.debug is not False:
    ml_options.append('-debug')
if args.explain is not None:
    ml_options.append('-explain')
    ml_options.append(args.explain)
if args.gpu is not None:
    ml_options.append('-gpu')
    ml_options.append(args.gpu)
if args.stats is not None:
    ml_options.append('-stats')
    ml_options.append(args.stats)

# Assign script file to name received from argparse module
script_file = args.f

# find the systemML root path which contains the bin folder, the script folder and the target folder
# tolerate path with spaces
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root_dir = os.path.dirname(script_dir)
user_dir = os.getcwd()

scripts_dir = join(project_root_dir, 'scripts')
build_dir = join(project_root_dir, 'target')
lib_dir = join(build_dir, 'lib')

systemml_jar = build_dir + os.sep + "SystemML.jar"
jcuda_jars = glob.glob(lib_dir + os.sep + "jcu*.jar")
target_jars = ','.join(jcuda_jars) # Include all JCuda Jars

log4j_properties_path = join(project_root_dir, 'conf', 'log4j.properties.template')

build_err_msg = 'You must build the project before running this script.'
build_dir_err_msg = 'Could not find target directory ' + build_dir + '. ' + build_err_msg

# check if the project had been built and the jar files exist
if not (exists(build_dir)):
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
if not (exists(systemml_config_path)):
    shutil.copyfile(systemml_template_config_path, systemml_config_path)
    print('... created ' + systemml_config_path)

# if SystemML-config.xml is provided as arguments
if args.config is None:
    systemml_config_path_arg = systemml_config_path
else:
    systemml_config_path_arg = args.config


# from http://stackoverflow.com/questions/1724693/find-a-file-in-python
def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return join(root, name)
    return None

# if the script file path was omitted, try to complete the script path
if not (exists(script_file)):
    script_file_name = abspath(script_file)
    script_file_found = find_file(script_file, scripts_dir)
    if script_file_found is None:
        print('Could not find DML script: ' + script_file)
        print_usage_and_exit()
    else:
        script_file = script_file_found
        print('DML Script:' + script_file)

default_conf = 'spark.driver.extraJavaOptions=-Dlog4j.configuration=file:{}'.format(log4j_properties_path)

# Backslash problem in windows.
if platform.system() == 'Windows':
    default_conf = default_conf.replace('\\', '//')

if args.conf is not None:
    conf = ' --conf '.join(args.conf + [default_conf])
else:
    conf = default_conf

cmd_spark = [spark_path, '--class', 'org.apache.sysml.api.DMLScript',
             '--master', args.master, '--driver-memory', args.driver_memory,
             '--num-executors', args.num_executors, '--executor-memory', args.executor_memory,
             '--executor-cores', args.executor_cores, '--conf', conf, '--jars', target_jars,
             systemml_jar]

cmd_system_ml = ['-config', systemml_config_path_arg,
                 '-exec', vars(args)['exec'], '-f', script_file, ' '.join(ml_options)]

cmd = cmd_spark + cmd_system_ml

return_code = os.system(' '.join(cmd))
# For debugging
# print(' '.join(cmd))

if return_code != 0:
    print('Failed to run SystemML. Exit code :' + str(return_code))
    print(' '.join(cmd))
