#!/usr/bin/env bash
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

################################################################################
##  File:  pypi-upload.sh
##  Desc:  Upload python artifacts to pypi.org
################################################################################

SELF=$(cd $(dirname $0) && pwd)

dry_run_flag=0
while getopts ":n" opt; do
  case $opt in
    n) dry_run_flag=1 ;;
    \?) error "Invalid option: $OPTARG" ;;
  esac
done


RELEASE_VERSION=
RELEASE_TAG=
read -p "RELEASE_VERSION : " RELEASE_VERSION
read -p "RELEASE_TAG : " RELEASE_TAG

BASE_DIR=$PWD # SystemDS top level folder

# Checkout the voted commit and build the publish distribution
git checkout $RELEASE_TAG

mvn clean package -P distribution

cd $BASE_DIR/src/main/python

# Steps:
# 1. update systemds/project_info.py with the new version
sed -i "s/$RELEASE_VERSION-SNAPSHOT/$RELEASE_TAG/" systemds/project_info.py

# 2. generate distribution archives
python3 create_python_dist.py

# 2a. check generated distribution files
python3 -m twine check dist/*

# 3. upload the distribution archives to testpypi/pypi
#    - For testing follow https://packaging.python.org/tutorials/packaging-projects/
#    - Note: for testing use command prompt in windows and 
#              use Edit->paste to paste the API token (https://pypi.org/help/#invalid-auth)
#            else, use `right click` for paste in the terminal.

# Dev:
# Test upload to test.pypi.org
# Credentials are
# username: __token__ 
# password: pypi-DU5y...

if [[ $dry_run_flag != 1 ]]; then
  python twine upload dist/*
else
  python -m twine upload --repository testpypi dist/*
fi


cd $BASE_DIR

exit
