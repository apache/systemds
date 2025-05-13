#!/bin/bash
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

# A script to execute the tests inside the docker container.

cd /github/workspace/src/main/cpp
./build.sh
cd /github/workspace

export MAVEN_OPTS="-Xmx512m"

log="/tmp/sysdstest.log"
mvn -ntp -B test-compile 2>&1 | grep -E "BUILD|Total time:|---|Building SystemDS"
mvn -ntp -B test -D maven.test.skip=false -D automatedtestbase.outputbuffering=true -D test=$1 2>&1 | grep -v "already exists in destination." | tee $log
mvn -ntp -B jacoco:merge

grep_args="SUCCESS"
grepvals="$( tail -n 100 $log | grep $grep_args)"

if [[ $grepvals == *"SUCCESS"* ]]; then
	exit 0
else
	exit 1
fi
