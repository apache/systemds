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

# error help print
printUsageExit()
{
cat << EOF
Usage: $0 <dml-filename> [arguments] [-help]
    -help     - Print this usage message and exit
Default Java options (-Xmx4g -Xms4g -Xmn400m) can be overridden by setting SYSTEMML_STANDALONE_OPTS.
EOF
  exit 1
}
#    Script internally invokes 'java [SYSTEMML_STANDALONE_OPTS] -jar StandaloneSystemML.jar -f <dml-filename> -exec singlenode -config=SystemML-config.xml [arguments]'

while getopts "h:" options; do
  case $options in
    h ) echo Warning: Help requested. Will exit after usage message;
        printUsageExit
        ;;
    \? ) echo Warning: Help requested. Will exit after usage message;
        printUsageExit
        ;;
    * ) echo Error: Unexpected error while processing options;
  esac
done

if [ -z $1 ] ; then
    echo "Wrong Usage.";
    printUsageExit;
fi

# Peel off first argument so that $@ contains arguments to DML script
SCRIPT_FILE=$1
shift

# Build up a classpath with all included libraries
CURRENT_PATH=$( cd $(dirname $0) ; pwd -P )

CLASSPATH=""
for f in ${CURRENT_PATH}/lib/*.jar; do
  CLASSPATH=${CLASSPATH}:$f;
done

LOG4JPROP=log4j.properties

# set default java opts if none supplied
if [ -z "$SYSTEMML_STANDALONE_OPTS" ] ; then
  SYSTEMML_STANDALONE_OPTS="-Xmx4g -Xms4g -Xmn400m"
fi;

# invoke the jar with options and arguments
CMD="\
java ${SYSTEMML_STANDALONE_OPTS} \
-cp ${CLASSPATH} \
-Dlog4j.configuration=file:${LOG4JPROP} \
org.apache.sysml.api.DMLScript \
-f ${SCRIPT_FILE} \
-exec singlenode \
-config=$CURRENT_PATH"/SystemML-config.xml" \
$@"

$CMD

# if there was an error, display the full java command
# RETURN_CODE=$?
# if [ $RETURN_CODE -ne 0 ]
# then
#   echo "Failed to run SystemML. Exit code: $RETURN_CODE"
#   echo ${CMD}
# fi
