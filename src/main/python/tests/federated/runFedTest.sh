#/bin/bash
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

# if [ "$#" -ne 0 ]; then
# 	echo "Usage:   "$0" <federatedTest>"
# 	echo "Example: "$0" tests/federated/test_federated_basic.py"
# 	exit
# fi

# FIELDS
workerdir="tests/federated/worker/"
outputdir="tests/federated/output/"
tmpfiledir="tests/federated/tmp/"
mkdir -p $workerdir
mkdir -p $outputdir
w1_Output="$workerdir/w1"
w2_Output="$workerdir/w2"
w3_Output="$workerdir/w3"
log="$outputdir/out.log"

# Make the workers start quietly and pipe their output to a file to print later
export SYSDS_QUIET=1
systemds WORKER 8001 >$w1_Output 2>&1 &
Fed1=$!
systemds WORKER 8002 >$w2_Output 2>&1 &
Fed2=$!
systemds WORKER 8003 >$w3_Output 2>&1 &
Fed3=$!
echo "Starting workers" && sleep 6 && echo "Starting tests"

# Run test
coverage run -m unittest discover -s tests/federated -p 'test_*.py' $1 >$log 2>&1
pkill -P $Fed1
pkill -P $Fed2
pkill -P $Fed3

# Print output
echo -e "\n---------------\nWorkers Output:\n---------------"
echo -e "\nWorker 1:"
cat $w1_Output
echo -e "\nWorker 2:"
cat $w2_Output
echo -e "\nWorker 3:"
cat $w3_Output
echo -e "\n------------\nTest output:\n------------"
cat $log
grepvals="$(tail -n 10 $log | grep OK)"
echo -e "------------\n"

# Cleanup
rm -r $workerdir
rm -r $outputdir
rm -r $tmpfiledir

if [[ $grepvals == *"OK"* ]]; then
	exit 0
else
	exit 1
fi
