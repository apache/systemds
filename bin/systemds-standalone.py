#!/usr/bin/env python3
#-------------------------------------------------------------
#
# Copyright 2019 Graz University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------

import os
import sys
from os.path import join
import argparse
import platform
from utils import get_env_systemds_root, find_dml_file, log4j_path, config_path


def default_classpath(systemds_root):
    """
    Classpath information required for excution

    return: String
    Classpath location of build, library and hadoop directories
    """
    build_lib = join(systemds_root, 'target', '*')
    lib_lib = join(systemds_root, 'target', 'lib', '*')
    hadoop_lib = join(systemds_root, 'target', 'lib', 'hadoop', '*')
    sysds_jar = join(systemds_root, 'target', 'classes')
    return build_lib, lib_lib, hadoop_lib, sysds_jar


def standalone_execution_entry(nvargs, args, config, explain, debug, stats, gpu, heapmem, f):
    """
    This function is responsible for the execution of arguments via
    subprocess call in singlenode mode
    """

    systemds_root = get_env_systemds_root()
    script_file = find_dml_file(systemds_root, f)

    if platform.system() == 'Windows':
        default_cp = ';'.join(default_classpath(systemds_root))
    else:
        default_cp = ':'.join(default_classpath(systemds_root))

    java_memory = '-Xmx' + heapmem + ' -Xms4g -Xmn1g'

    # Log4j
    log4j = log4j_path(systemds_root)
    log4j_properties_path = '-Dlog4j.configuration=file:{}'.format(log4j)

    # Config
    if config is None:
        default_config = config_path(systemds_root)
    else:
        default_config = config

    ds_options = []
    if nvargs is not None:
        ds_options.append('-nvargs')
        ds_options.append(' '.join(nvargs))
    if args is not None:
        ds_options.append('-args')
        ds_options.append(' '.join(args))
    if explain is not None:
        ds_options.append('-explain')
        ds_options.append(explain)
    if debug is not False:
        ds_options.append('-debug')
    if stats is not None:
        ds_options.append('-stats')
        ds_options.append(stats)
    if gpu is not None:
        ds_options.append('-gpu')
        ds_options.append(gpu)

    os.environ['HADOOP_HOME'] = '/tmp/systemds'
    
    cmd = ['java', java_memory, log4j_properties_path,
           '-cp', default_cp, 'org.tugraz.sysds.api.DMLScript',
           '-f', script_file, '-exec', 'singlenode', '-config', default_config,
           ' '.join(ds_options)]

    cmd = ' '.join(cmd)
    print(cmd)

    return_code = os.system(cmd)
    return return_code


if __name__ == '__main__':

    fn = sys.argv[0]
    if os.path.exists(fn):
        #print(os.path.basename(fn))
        print(fn[:fn.rfind('/')])
    
    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      description='System-DS Standalone Script')

    # SYSTEM-DS Options
    cparser.add_argument('-nvargs', help='List of attributeName-attributeValue pairs', nargs='+', metavar='')
    cparser.add_argument('-args', help='List of positional argument values', metavar='', nargs='+')
    cparser.add_argument('-config', help='System-DS configuration file (e.g SystemDS-config.xml)', metavar='')
    cparser.add_argument('-explain', help='explains plan levels can be hops, runtime, '
                                          'recompile_hops, recompile_runtime', nargs='?', const='runtime', metavar='')
    cparser.add_argument('-debug', help='runs in debug mode', action='store_true')
    cparser.add_argument('-stats', help='Monitor and report caching/recompilation statistics, '
                                        'heavy hitter <count> is 10 unless overridden', nargs='?', const='10',
                         metavar='')
    cparser.add_argument('-gpu', help='uses CUDA instructions when reasonable, '
                                      'set <force> option to skip conservative memory estimates '
                                      'and use GPU wherever possible', nargs='?')
    cparser.add_argument('-heapmem', help='maximum JVM heap memory', metavar='', default='8g')
    cparser.add_argument('-f', required=True, help='specifies dml file to execute; '
                                                   'path can be local/hdfs/gpfs', metavar='')

    args = cparser.parse_args()
    arg_dict = vars(args)
    return_code = standalone_execution_entry(**arg_dict)

    if return_code != 0:
        print('Failed to run SystemDS. Exit code :' + str(return_code))
