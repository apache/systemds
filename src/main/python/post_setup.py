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

from __future__ import print_function
import os
import sys
import platform

try:
    exec(open('systemml/project_info.py').read())
except IOError:
    print("Could not read project_info.py.", file=sys.stderr)
    sys.exit
ARTIFACT_NAME = __project_artifact_id__
ARTIFACT_VERSION = __project_version__
ARTIFACT_VERSION_SHORT = ARTIFACT_VERSION.split("-")[0]

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
if platform.system() == "Windows":
    os.rename(
        os.path.join(root_dir, 'target', ARTIFACT_NAME + '-' + ARTIFACT_VERSION_SHORT + '.zip'),
        os.path.join(root_dir, 'target', ARTIFACT_NAME + '-' + ARTIFACT_VERSION + '-python.zip'))
else:
    os.rename(
        os.path.join(root_dir, 'target', ARTIFACT_NAME + '-' + ARTIFACT_VERSION_SHORT + '.tar.gz'),
        os.path.join(root_dir, 'target', ARTIFACT_NAME + '-' + ARTIFACT_VERSION + '-python.tgz'))
