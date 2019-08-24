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

import sys
import os
from os.path import join, exists
from os import environ
import shutil


def get_env_systemds_root():
    """
    Env variable error check and path location

    return: String
    Location of SYSTEMDS_ROOT
    """
    systemds_root = os.environ.get('SYSTEMDS_ROOT')
    if systemds_root is None:
        #print('SYSTEMDS_ROOT not found')
        #sys.exit()
        fn = sys.argv[0]
        systemds_root = fn[:fn.rfind('/')] + '/..'

    return systemds_root


def get_env_spark_root():
    """
    Env variable error check and path location

    return: String
    Location of SPARK_ROOT
    """
    spark_root = environ.get('SPARK_ROOT')
    if spark_root is None:
        print('SPARK_ROOT not found')
        sys.exit()

    return spark_root


def find_file(name, path):
    """
    Responsible for finding a specific file recursively given a location
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return join(root, name)


def find_dml_file(systemds_root, script_file):
    """
    Find the location of DML script being executed

    return: String
    Location of the dml script
    """
    scripts_dir = join(systemds_root, 'scripts')
    if not exists(script_file):
        script_file_path = find_file(script_file, scripts_dir)
        if script_file_path is not None:
            return script_file_path
        else:
            print('Could not find DML script: ' + script_file)
            sys.exit()

    return script_file

def log4j_path(systemds_root):
    """
    Create log4j.properties from the template if not exist

    return: String
    Location of log4j.properties path
    """
    log4j_properties_path = join(systemds_root, 'conf', 'log4j.properties')
    log4j_template_properties_path = join(systemds_root, 'conf', 'log4j.properties.template')
    if not (exists(log4j_properties_path)):
        shutil.copyfile(log4j_template_properties_path, log4j_properties_path)
        print('... created ' + log4j_properties_path)
    return log4j_properties_path


def config_path(systemds_root):
    """
    Create SystemDS-config from the template if not exist

    return: String
    Location of SystemDS-config.xml
    """
    systemds_config_path = join(systemds_root, 'conf', 'SystemDS-config.xml')
    systemds_template_config_path = join(systemds_root, 'conf', 'SystemDS-config.xml.template')
    if not (exists(systemds_config_path)):
        shutil.copyfile(systemds_template_config_path, systemds_config_path)
        print('... created ' + systemds_config_path)
    return systemds_config_path
