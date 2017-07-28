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
from os.path import join, exists
from os import environ


def get_env():
    systemml_home = os.environ.get('SYSTEMML_HOME')
    if systemml_home is None:
        print('SYSTEMML_HOME not found')
        sys.exit()
    spark_home = environ.get('SPARK_HOME')
    if spark_home is None:
        print('SPARK_HOME not found')
        sys.exit()

    return spark_home, systemml_home


def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return join(root, name)
    return None


def find_script_file(systemml_home, script_file):
    scripts_dir = join(systemml_home, 'scripts')
    if not (exists(script_file)):
        script_file = find_file(script_file, scripts_dir)
        if script_file is None:
            print('Could not find DML script: ' + script_file)
            sys.exit()

    return script_file


def log4j_path(systemml_home):
    log4j_properties_path = join(systemml_home, 'conf', 'log4j.properties')
    log4j_template_properties_path = join(systemml_home, 'conf', 'log4j.properties.template')
    if not (exists(log4j_properties_path)):
        shutil.copyfile(log4j_template_properties_path, log4j_properties_path)
        print('... created ' + log4j_properties_path)
    return log4j_properties_path


def config_path(systemml_home):
    systemml_config_path = join(systemml_home, 'conf', 'SystemML-config.xml')
    systemml_template_config_path = join(systemml_home, 'conf', 'SystemML-config.xml.template')
    if not (exists(systemml_config_path)):
        shutil.copyfile(systemml_template_config_path, systemml_config_path)
        print('... created ' + systemml_config_path)
    return systemml_config_path

