#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

function exit_with_usage {
  cat << EOF

release-verify - Verifies the artifacts from a given directory.

SYNOPSIS

usage: release-verify.sh [--compile | --verifyAll | verifyLic | verifyBin]

DESCRIPTION

Once artifacts are generated, this utility will verify the artifacts.

--compile
This will compile the utility source code which is not on regular source code path.

--verifyAll <--tag="Code based on Git tag will be validated."> [--workDir="Directory where output files will be created]
This will verify license, notice and binary files. 

--verifyLic [--distDir="Directory Containing zip/tgz files]
This will verify license, notice in zip/tgz files. 

--verifyBin <--tag="Code based on Git tag will be validated."> [--workDir="Directory where output files will be created]
This will verify binary distribution files for runtime correctness. 

OPTIONS

--distDir            - Directory containing release artifacts (zip/tzg) files.

--workDir             - Directory where output files will be created.

EXAMPLES

# To compile utility source code
release-verify.sh --compile

# To verify release artifacts
release-verify.sh --verifyAll --tag=<tagName>
e.g. ./release-verify.sh --verifyAll --tag=v0.14.0-rc4

# To verify license and notices
release-verify.sh --verifyLic --distDir=<DistribLocation>
e.g. ./release-verify.sh --verifyLic --distDir=tmp/relValidation/downloads

# To verify binary files
release-verify.sh --verifyBin --tag=<tagName>
e.g. ./release-verify.sh --verifyBin --tag=v0.14.0-rc4


EOF
  exit 1
}

set -e

if [ $# -eq 0 ]; then
  echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Insuffient parameters passed.";
  exit_with_usage
fi


# Process each provided argument configuration
while [ "${1+defined}" ]; do
  IFS="=" read -ra PARTS <<< "$1"
  case "${PARTS[0]}" in
    --compile)
      COMPILE_CODE=true
      shift
      ;;
    --verifyAll)
      LIC_NOTICE_VERIFY=true
      BIN_VERIFY=true
      shift
      ;;
    --verifyLic)
      LIC_NOTICE_VERIFY=true
      shift
      ;;
    --verifyBin)
      BIN_VERIFY=true
      shift
      ;;
    --distDir)
      DIST_DIR="${PARTS[1]}"
      shift
      ;;
    --workDir)
      WORK_DIR="${PARTS[1]}"
      shift
      ;;
    --tag)
      TAG="${PARTS[1]}"
      shift
      ;;
    *help* | -h)
      exit_with_usage
     exit 0
     ;;
    -*)
     echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Unknown option: $1" >&2
     exit_with_usage
     ;;
    *)  # No more options
     break
     ;;
  esac
done

ORIG_DIR=$(pwd)
EXEC_DIR="`dirname \"$0\"`"
if [[ ${EXEC_DIR:0:1} != "/" ]]; then
    EXEC_DIR=$ORIG_DIR/$EXEC_DIR
fi

if [[ -z "$WORK_DIR" ]]; then
    WORK_DIR="$EXEC_DIR/tmp/relValidation"
elif [[ ${WORK_DIR:0:1} != "/" ]]; then
    WORK_DIR="$ORIG_DIR/$WORK_DIR"
fi

# If --verifyAll has been specified en license and notice validation should be done from place where all files downloaded in --verifyBin step.
if [[ "$BIN_VERIFY" == "true" && "$LIC_NOTICE_VERIFY" == "true" ]]; then
    DIST_DIR=$WORK_DIR/downloads
fi

if [[ "$BIN_VERIFY" == "true" && -z "$TAG" ]]; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Argument --tag is a mandatory variable for binary verification."
    exit_with_usage
fi

if [[ "$LIC_NOTICE_VERIFY" == "true" && -z "$DIST_DIR" ]]; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: WARNING: Since --distDir has not passed, default distribution directory '$EXEC_DIR/target/release/systemml/target' has been used."
    DIST_DIR="$EXEC_DIR/target/release/systemml/target"
elif [[ ${DIST_DIR:0:1} != "/" ]]; then
    DIST_DIR="$ORIG_DIR/$DIST_DIR"
fi

cd $EXEC_DIR/src/test/java
if [[ "$COMPILE_CODE" == "true" ]]; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Compiling release verify utility..."

    javac -Xlint:unchecked -classpath ../../../../..//target/lib/commons-compress-1.4.1.jar:../../../../..//target/lib/commons-io-2.4.jar:. org/apache/sysml/validation/*.java

    cd "$ORIG_DIR" # Return to directoryt from it was called.
    exit 0
fi

if [[ "$BIN_VERIFY" == "true" ]]; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying binary files for runtime execution..."

    $EXEC_DIR/src/test/bin/verifyBuild.sh $TAG $WORK_DIR
    RET_CODE=$?
    if [[ $RET_CODE == 0 ]]; then
       echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verification of binary files for runtime execution completed successfully."
    else
       echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Verification of binary files for runtime execution failed."
       cd $ORIG_DIR
       exit $RET_CODE
    fi
    echo "*********************************************************************************************************************"
fi

if [[ "$LIC_NOTICE_VERIFY" == "true" ]]; then
    echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verifying license and notices from zip/tgz files..."

    java -classpath ../../../../..//target/lib/commons-compress-1.4.1.jar:../../../../..//target/lib/commons-io-2.4.jar:. org/apache/sysml/validation/ValidateLicAndNotice $DIST_DIR
    RET_CODE=$?
    if [[ $RET_CODE == 0 ]]; then
       echo "`date +%Y-%m-%dT%H:%M:%S`: INFO: Verification of license and notices completed successfully."
    else
       echo "`date +%Y-%m-%dT%H:%M:%S`: ERROR: Verification of license and notices failed."
    fi
    echo "*********************************************************************************************************************"
fi

cd "$ORIG_DIR" # Return to directory from it was called.
exit $RET_CODE
