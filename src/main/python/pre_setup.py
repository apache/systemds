#!/usr/bin/env python3
# -------------------------------------------------------------
#
# Modifications Copyright 2020 Graz University of Technology
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
import shutil
import fnmatch

python_dir = 'systemds'
java_dir = 'systemds-java'
java_dir_full_path = os.path.join(python_dir, java_dir)
if os.path.exists(java_dir_full_path):
    shutil.rmtree(java_dir_full_path, True)
os.mkdir(java_dir_full_path)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
JAR_FILE_NAMES = ['systemds-*-SNAPSHOT.jar', 'systemds-*.jar', 'systemds-*-SNAPSHOT-extra', 'systemds-*-extra.jar']
EXCLUDE_JAR_FILE_NAMES = ['systemds-*javadoc.jar', 'systemds-*sources.jar', 'systemds-*standalone.jar',
                          'systemds-*lite.jar']
for file in os.listdir(os.path.join(root_dir, 'target')):
    if any((fnmatch.fnmatch(file, valid_name) for valid_name in JAR_FILE_NAMES)) and not any(
            (fnmatch.fnmatch(file, exclude_name) for exclude_name in EXCLUDE_JAR_FILE_NAMES)):
        shutil.copyfile(os.path.join(root_dir, 'target', file), os.path.join(java_dir_full_path, file))
    if file == 'lib':
        shutil.copytree(os.path.join(root_dir, 'target', file), os.path.join(java_dir_full_path, file))

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
shutil.copyfile(os.path.join(root_dir, 'LICENSE'), 'LICENSE')
shutil.copyfile(os.path.join(root_dir, 'NOTICE'), 'NOTICE')
