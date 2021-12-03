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

import os
import shutil
import fnmatch
from zipfile import ZipFile

this_path = os.path.dirname(os.path.realpath(__file__))
PYTHON_DIR = 'systemds'

# Go three directories out this is the root dir of systemds repository
SYSTEMDS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(this_path))) 

# temporary directory for unzipping of bin zip
TMP_DIR = os.path.join(this_path, 'tmp')
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR, True)
os.mkdir(TMP_DIR)


# Copy jar files from release artifact.
LIB_DIR = os.path.join(this_path, PYTHON_DIR, 'lib')
if os.path.exists(LIB_DIR):
    shutil.rmtree(LIB_DIR, True)
SYSTEMDS_BIN = 'systemds-*-bin.zip'
found_bin = False
for file in os.listdir(os.path.join(SYSTEMDS_ROOT, 'target')):
    # Take jar files from bin release file
    if fnmatch.fnmatch(file, SYSTEMDS_BIN):
        if found_bin:
            print("invalid install found multiple bin files, please package systemds with clean flag")
            exit(-1)
        found_bin = True

for file in os.listdir(os.path.join(SYSTEMDS_ROOT, 'target')):
    # Take jar files from bin release file
    if fnmatch.fnmatch(file, SYSTEMDS_BIN):
        print("Using java files from : " + file )
        systemds_bin_zip = os.path.join(SYSTEMDS_ROOT, 'target', file)
        extract_dir = os.path.join(TMP_DIR)

        with ZipFile(systemds_bin_zip, 'r') as zip:
            for f in zip.namelist():
                split_path = os.path.split(os.path.dirname(f))
                if split_path[1] == 'lib':
                    zip.extract(f, TMP_DIR)
        unzipped_dir_name = file.rsplit('.', 1)[0]
        shutil.copytree(os.path.join(TMP_DIR, unzipped_dir_name, 'lib'), LIB_DIR)
        break

# Take hadoop binaries.
HADOOP_DIR_SRC = os.path.join(SYSTEMDS_ROOT, 'target', 'lib', 'hadoop')
if os.path.exists(HADOOP_DIR_SRC):
    shutil.copytree(HADOOP_DIR_SRC, os.path.join(LIB_DIR,"hadoop"))

# Take conf files.
CONF_DIR = os.path.join(this_path, PYTHON_DIR, 'conf')
if not os.path.exists(CONF_DIR):
    os.mkdir(CONF_DIR)
shutil.copy(os.path.join(SYSTEMDS_ROOT,'conf', 'log4j.properties'), os.path.join(this_path, PYTHON_DIR, 'conf', 'log4j.properties'))
shutil.copy(os.path.join(SYSTEMDS_ROOT,'conf', 'SystemDS-config-defaults.xml'), os.path.join(this_path, PYTHON_DIR, 'conf', 'SystemDS-config-defaults.xml'))

SYSTEMDS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
shutil.copyfile(os.path.join(SYSTEMDS_ROOT, 'LICENSE'), 'LICENSE')
shutil.copyfile(os.path.join(SYSTEMDS_ROOT, 'NOTICE'), 'NOTICE')

# Remove old build and dist path
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR, True)
build_path = os.path.join(this_path, 'build')
if os.path.exists(build_path):
    shutil.rmtree(build_path, True)
dist_path = os.path.join(this_path, 'dist')
if os.path.exists(dist_path):
    shutil.rmtree(dist_path, True)
egg_path = os.path.join(this_path, 'systemds.egg-info')
if os.path.exists(egg_path):
    shutil.rmtree(egg_path, True)
