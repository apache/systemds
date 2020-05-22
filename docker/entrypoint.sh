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

cd /github/workspace

build="$(mvn -T 2 clean compile test-compile | grep 'BUILD')"

if [[ $build == *"SUCCESS"* ]]; then
  echo "Successfull build"
else
  echo "failed building"
  exit 1
fi

log="/tmp/sysdstest.log"

echo "Starting Tests"

mvn surefire:test -DskipTests=false -Dtest=$1 2>&1 > $log

grep_args="SUCCESS"
grepvals="$( tail -n 100 $log | grep $grep_args)"

if [[ $grepvals == *"SUCCESS"* ]]; then
	echo "--------------------- last 100 lines from test ------------------------"
	tail -n 100 $log
	echo "------------------ last 100 lines from test end -----------------------"
	sleep 3
	exit 0
else
	echo "\n $(cat $log)"
	sleep 3
	exit 1
fi
