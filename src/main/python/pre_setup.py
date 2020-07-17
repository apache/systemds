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
