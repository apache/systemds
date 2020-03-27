#!/usr/bin/env bash
#-------------------------------------------------------------
#
# Modifications Copyright 2019 Graz University of Technology
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
Usage: $0 <Tag Name> [Working Directory] [-help]
    Tag Name:           is the name of tag for which verification will be done
    Working Directory:  will be used to clone/download repo/files for validation
    -help:              Print this usage message and exit
EOF
  exit 1
}

runCommand()
{
    command=$1
    echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Running command '$command' ..." >>$OUT_FILE
    eval ${command} 1>>$OUT_FILE 2>>$ERR_FILE

    RETURN_CODE=$?

    # if there was an error, display the full command
    if [ $RETURN_CODE -ne 0 ]
    then
        echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Failed to run command '$command' with exit code: $RETURN_CODE"
        echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Failed to run command '$command' with exit code: $RETURN_CODE" >>$OUT_FILE
        LF=$'\n'
        exit 1
    fi
}

while getopts "h:" options; do
  case $options in
    h ) echo Warning: Help requested. Will exit after usage message
        printUsageExit
        ;;
    \? ) echo Warning: Help requested. Will exit after usage message
        printUsageExit
        ;;
    * ) echo Error: Unexpected error while processing options
  esac
done

if [ -z "$1" ] ; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Insufficient parameters passed."; # TagName has not passed.
    printUsageExit;
fi

if [ -z "$SPARK_HOME" ] ; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Environment variable 'SPARK_HOME' has not been defined.";
    printUsageExit;
fi

if [ -z "$HADOOP_HOME" ] ; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Environment variable 'HADOOP_HOME' has not been defined.";
    printUsageExit;
fi

# find the root path which contains the script file
# tolerate path with spaces
SCRIPT_DIR=$( dirname "$0" )
USER_DIR=$( cd "${SCRIPT_DIR}/../../../" ; pwd -P )

TAG_NAME=$1
shift
WORKING_DIR=$1

if [ -z $WORKING_DIR ] ; then
    WORKING_DIR="$USER_DIR/tmp/relValidation"
fi

rm -rf "$WORKING_DIR"/systemds
mkdir -p "$WORKING_DIR"
OUT_FILE=$WORKING_DIR/relValidation.out
ERR_FILE=$WORKING_DIR/relValidation.err

## Clone the branch and build distribution package
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Cloning branch and building distribution package..."
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Cloning branch and building distribution package." >> $OUT_FILE
echo "=========================================================================================================" >> $OUT_FILE
runCommand "cd $WORKING_DIR"
runCommand "git clone https://github.com/tugraz-isds/systemds.git"
runCommand "cd systemds"
runCommand "git checkout tags/$TAG_NAME -b $TAG_NAME"
runCommand "mvn -Dmaven.repo.local=$HOME/.m2/temp-repo clean package -P distribution"
echo "=========================================================================================================" >> $OUT_FILE

# Removes v from tag to get distribution directory name
if [[ ${TAG_NAME:0:1} == "v" ]]; then
    DIST_DIR=${TAG_NAME:1}
else
    DIST_DIR=$TAG_NAME
fi

VER_NAME=`expr "$DIST_DIR" : '^\(.[.0-9]*\)'`

## Download binaries from distribution location and verify them
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Downloading binaries from distribution location..."
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Downloading binaries from distribution location." >> $OUT_FILE
runCommand "mkdir -p $WORKING_DIR/downloads"
runCommand "cd $WORKING_DIR/downloads"
#ToDo: release staging location
#runCommand "wget -r -nH -nd -np -R 'index.html*' https://dist.apache.org/repos/dist/dev/systemml/$DIST_DIR/"
echo "=========================================================================================================" >> $OUT_FILE

## Verify binary tgz files
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying binary tgz files..."
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying binary tgz files..." >> $OUT_FILE
runCommand "rm -rf systemds-$VER_NAME-bin"
runCommand "tar -xvzf systemds-$VER_NAME-bin.tgz"
runCommand "cd systemds-$VER_NAME-bin"
runCommand "echo \"print('hello world');\" > hello.dml"
runCommand "./systemds-standalone.sh hello.dml"
runCommand "cd .."

## Verify binary zip files
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying binary zip files..."
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying binary zip files..." >> $OUT_FILE
runCommand "rm -rf systemds-$VER_NAME-bin"
runCommand "unzip systemds-$VER_NAME-bin.zip"
runCommand "cd systemds-$VER_NAME-bin"
runCommand "echo \"print('hello world');\" > hello.dml"
runCommand "./systemds-standalone.sh hello.dml"
runCommand "cd .."

## Verify src tgz files
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying source tgz files..."
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying source tgz files..." >> $OUT_FILE
runCommand "rm -rf systemds-$VER_NAME-src"
runCommand "tar -xvzf systemds-$VER_NAME-src.tgz"
runCommand "cd systemds-$VER_NAME-src"
runCommand "mvn clean package -P distribution"
runCommand "cd target"
runCommand "java -cp \"./lib/*:systemds-$VER_NAME.jar\" org.tugraz.sysds.api.DMLScript -s \"print('hello world');\""
runCommand "java -cp \"./lib/*:SystemDS.jar\" org.tugraz.sysds.api.DMLScript -s \"print('hello world');\""
runCommand "cd ../.."

## Verify Spark batch mode
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying Spark batch mode..."
echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying Spark batch mode..." >> $OUT_FILE
runCommand "cd systemds-$VER_NAME-bin/lib"
runCommand "$SPARK_HOME/bin/spark-submit systemds-$VER_NAME.jar -s \"print('hello world');\" -exec hybrid"

echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verification of binary files completed successfully."
# echo "================================================================================"

exit 0
