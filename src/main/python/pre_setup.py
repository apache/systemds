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
python_dir = 'systemds'
java_dir = 'systemds-java'
java_dir_full_path = os.path.join(this_path, python_dir, java_dir)
if os.path.exists(java_dir_full_path):
    shutil.rmtree(java_dir_full_path, True)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))

# temporary directory for unzipping of bin zip
TMP_DIR = os.path.join(this_path, 'tmp')
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR, True)
os.mkdir(TMP_DIR)

SYSTEMDS_BIN = 'systemds-*-SNAPSHOT-bin.zip'
for file in os.listdir(os.path.join(root_dir, 'target')):
    if fnmatch.fnmatch(file, SYSTEMDS_BIN):
        new_path = os.path.join(TMP_DIR, file)
        shutil.copyfile(os.path.join(root_dir, 'target', file), new_path)
        extract_dir = os.path.join(TMP_DIR)
        with ZipFile(new_path, 'r') as zip:
            for f in zip.namelist():
                split_path = os.path.split(os.path.dirname(f))
                if split_path[1] == 'lib':
                    zip.extract(f, TMP_DIR)
        unzipped_dir_name = file.rsplit('.', 1)[0]
        shutil.copytree(os.path.join(TMP_DIR, unzipped_dir_name), java_dir_full_path)
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR, True)

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
shutil.copyfile(os.path.join(root_dir, 'LICENSE'), 'LICENSE')
shutil.copyfile(os.path.join(root_dir, 'NOTICE'), 'NOTICE')

# delete old build and dist path
build_path = os.path.join(this_path, 'build')
if os.path.exists(build_path):
    shutil.rmtree(build_path, True)
dist_path = os.path.join(this_path, 'dist')
if os.path.exists(dist_path):
    shutil.rmtree(dist_path, True)
