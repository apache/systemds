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
import glob
from os.path import join, exists, abspath
import argparse
from utils import get_env, find_script_file, log4j_path, config_path


def default_jars(systemml_home):
    build_dir = join(systemml_home, 'target')
    lib_dir = join(build_dir, 'lib')
    systemml_jar = build_dir + os.sep + "SystemML.jar"
    jcuda_jars = glob.glob(lib_dir + os.sep + "jcu*.jar")
    target_jars = ','.join(jcuda_jars)
    return target_jars, systemml_jar


# TODO:
# Test config
# add comments
# Argparse add default and options
# Windows Support
def spark_submit_entry(master, driver_memory, num_executors, executor_memory,
                       executor_cores, conf,
                       nvargs, args, config, explain, debug, stats, gpu, f):

    spark_home, systemml_home = get_env()
    spark_path = join(spark_home, 'bin', 'spark-submit')
    script_file = find_script_file(systemml_home, f)

    # Jars
    cuda_jars, systemml_jars = default_jars(systemml_home)

    # Log4j
    log4j = log4j_path(systemml_home)
    log4j_properties_path = 'spark.driver.extraJavaOptions=-Dlog4j.configuration=file:{}'.format(log4j)
    if conf is None:
        default_conf = log4j_properties_path
    else:
        default_conf = ' --conf '.join(conf + [default_conf])

    # Config
    if config is None:
        default_config = config_path(systemml_home)
    else:
        default_config = ' --conf '.join(conf + [default_conf])

    #  optional arguments
    ml_options = []
    if nvargs is not None:
        ml_options.append('-nvargs')
        ml_options.append(' '.join(nvargs))
    if args is not None:
        ml_options.append('-args')
        ml_options.append(' '.join(args))
    if explain is not None:
        ml_options.append('-explain')
        ml_options.append(explain)
    if debug is not False:
        ml_options.append('-debug')
    if stats is not None:
        ml_options.append('-stats')
        ml_options.append(stats)
    if gpu is not None:
        ml_options.append('-gpu')
        ml_options.append(gpu)

    if len(ml_options) < 1:
        ml_options = ''

    # stats, explain, target_jars
    cmd_spark = [spark_path, '--class', 'org.apache.sysml.api.DMLScript',
                 '--master', master, '--driver-memory', driver_memory,
                 '--num-executors', num_executors, '--executor-memory', executor_memory,
                 '--executor-cores', executor_cores, '--conf', default_conf,
                 '--jars', cuda_jars, systemml_jars]

    cmd_system_ml = ['-config', default_config,
                     '-exec', 'hybrid_spark', '-f', script_file, ' '.join(ml_options)]

    cmd = cmd_spark + cmd_system_ml
    return_code = os.system(' '.join(cmd))
    return return_code


if __name__ == '__main__':
    spark_home, systemml_home = get_env()

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
    cparser.add_argument('-explain', help='explains plan levels can be hops, runtime, '
                                          'recompile_hops, recompile_runtime', nargs='?', const='runtime', metavar='')
    cparser.add_argument('-debug', help='runs in debug mode', action='store_true')
    cparser.add_argument('-stats', help='Monitor and report caching/recompilation statistics, '
                                        'heavy hitter <count> is 10 unless overridden', nargs='?', const='10',
                                   metavar='')
    cparser.add_argument('-gpu', help='uses CUDA instructions when reasonable, '
                                      'set <force> option to skip conservative memory estimates '
                                      'and use GPU wherever possible', nargs='?')
    cparser.add_argument('-f', required=True, help='specifies dml/pydml file to execute; '
                                                   'path can be local/hdfs/gpfs', metavar='')

    args = cparser.parse_args()
    arg_dict = vars(args)

    return_code = spark_submit_entry(**arg_dict)

    if return_code != 0:
        print('Failed to run SystemML. Exit code :' + str(return_code))
