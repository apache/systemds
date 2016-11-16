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

import os, shutil
python_dir = 'systemml'
java_dir='systemml-java'
java_dir_full_path = os.path.join(python_dir, java_dir)
if os.path.exists(java_dir_full_path):
    shutil.rmtree(java_dir_full_path, True)
os.mkdir(java_dir_full_path)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
shutil.copyfile(os.path.join(root_dir, 'target', 'SystemML.jar'), os.path.join(java_dir_full_path, 'SystemML.jar'))
shutil.copytree(os.path.join(root_dir, 'scripts'), os.path.join(java_dir_full_path, 'scripts'))