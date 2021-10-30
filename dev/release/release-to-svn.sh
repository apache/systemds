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
##  Desc:  Promote release candidate from dev/systemds to release/systemds
################################################################################

SELF=$(cd $(dirname $0) && pwd)


RELEASE_STAGING_LOCATION="https://dist.apache.org/repos/dist/dev/systemds"
RELEASE_LOCATION="https://dist.apache.org/repos/dist/dev/systemds"

RELEASE_VERSION=2.2.0
APPROVED_RELEASE_TAG=2.2.0-rc1

svn co $RELEASE_STAGING_LOCATION svn-dev-systemds
svn co --depth=empty $RELEASE_LOCATION svn-release-systemds
mkdir -p svn-release-systemds/$RELEASE_VERSION

cp svn-dev-systemds/${APPROVED_RELEASE_TAG}/systemds-* svn-release-systemds/$RELEASE_VERSION

svn add svn-release-systemds/$RELEASE_VERSION

cd svn-release-systemds
svn ci --username "$ASF_USERNAME" --password "$ASF_PASSWORD" -m"Apache SystemDS $RELEASE_VERSION Released" --no-auth-cache
