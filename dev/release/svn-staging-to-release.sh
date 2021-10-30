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
##  File:  release-to-svn.sh
##  Desc:  Promote release candidate from svn dev/systemds to release/systemds
##  Note:  This file to be run only after the succesful voting
################################################################################

SELF=$(cd $(dirname $0) && pwd)

dry_run_flag=0
while getopts ":n" opt; do
  case $opt in
    n) dry_run_flag=1 ;;
    \?) error "Invalid option: $OPTARG" ;;
  esac
done


RELEASE_STAGING_LOCATION="https://dist.apache.org/repos/dist/dev/systemds"
RELEASE_LOCATION="https://dist.apache.org/repos/dist/release/systemds"

RELEASE_VERSION=
APPROVED_RELEASE_TAG=

ASF_USERNAME=

read -p "RELEASE_VERSION : " RELEASE_VERSION
read -p "APPROVED_RELEASE_TAG : " APPROVED_RELEASE_TAG

read -p "ASF_USERNAME : " ASF_USERNAME

tmp_repo=$(mktemp -d systemds-repo-tmp-XXXXX)

pushd "${tmp_repo}"

# 1. Checkout only the directory associated with approved release tag
svn co --depth=empty $RELEASE_STAGING_LOCATION svn-dev-systemds
cd svn-dev-systemds
svn update --set-depth files ${APPROVED_RELEASE_TAG}
cd ..

# 2.1. Checkout the empty repo, and copy the contents from svn dev
svn co --depth=empty $RELEASE_LOCATION svn-release-systemds
mkdir -p svn-release-systemds/$RELEASE_VERSION

cp svn-dev-systemds/${APPROVED_RELEASE_TAG}/systemds-* svn-release-systemds/$RELEASE_VERSION

# 2.2. Add the files to svn
svn add svn-release-systemds/$RELEASE_VERSION
cd svn-release-systemds

# 2.3. Commit and upload the files to the svn repository

if [[ $dry_run_flag != 1 ]]; then
  # This step prompts for the Apache Credentials
  svn ci --username $ASF_USERNAME -m'Apache SystemDS $RELEASE_VERSION Released' --no-auth-cache \n
  [[ $? == 0 ]] && printf '\n Publishing to $RELEASE_LOCATION is complete!\n'
else
  printf "\n==========\n"
  printf "This step would commit to the SVN release repo\n"
  printf "At $RELEASE_LOCATION \n"
  printf "\n==========\n"
  printf "You might want to manually check the files and run the following:\n"
  printf "svn ci --username $ASF_USERNAME -m'Apache SystemDS $RELEASE_VERSION Released' --no-auth-cache \n"
  printf "\n==========\n"
fi


