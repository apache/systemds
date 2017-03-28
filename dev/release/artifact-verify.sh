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

artifact-verify - Verifies the artifacts from a given directory.

SYNOPSIS

usage: artifact-verify.sh [--compile | --verify]

DESCRIPTION

Once artifacts are generated, this utility will verify the artifacts.

--compile
This will compile the utility source code which is not on regular source code path.

--verify [--distDir="Directory Containing zip/tgz files]
Publish the maven artifacts of a release to the Apache staging maven repository.

OPTIONS

--distDir            - Directory containing release artifacts (zip/tzg) files.


EXAMPLES

# To compile utility source code
artifact-verify.sh --compile

# To verify release artifacts
artifact-verify.sh --verify --distDir="../../../target/release/incubator-systemml/target"

EOF
  exit 1
}

set -e

if [ $# -eq 0 ]; then
  exit_with_usage
fi


# Process each provided argument configuration
while [ "${1+defined}" ]; do
  IFS="=" read -ra PARTS <<< "$1"
  case "${PARTS[0]}" in
    --compile)
      GOAL="compile"
      COMPILE_CODE=true
      shift
      ;;
    --verify)
      GOAL="verify"
      ARTIFACT_VERIFY=true
      shift
      ;;
    --distDir)
      DIST_DIR="${PARTS[1]}"
      shift
      ;;

    *help* | -h)
      exit_with_usage
     exit 0
     ;;
    -*)
     echo "Error: Unknown option: $1" >&2
     exit 1
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
cd $EXEC_DIR/src/test/java

if [[ "$ARTIFACT_VERIFY" == "true" && -z "$DIST_DIR" ]]; then
    echo "WARNING: Since --distDir has not passed, default distribution directory '$EXEC_DIR/target/release/incubator-systemml/target' has been used."
    DIST_DIR="$EXEC_DIR/target/release/incubator-systemml/target"
elif [[ ${DIST_DIR:0:1} != "/" ]]; then
    DIST_DIR="$ORIG_DIR/$DIST_DIR"
fi

if [[ "$COMPILE_CODE" == "true" ]]; then
    echo "Compiling artifact utility..."

    javac -classpath ../../../../..//target/lib/commons-compress-1.4.1.jar:../../../../..//target/lib/commons-io-2.4.jar:. org/apache/sysml/validation/ValidateLicAndNotice.java

    cd "$ORIG_DIR" # Return to directoryt from it was called.
    exit 0
fi


if [[ "$ARTIFACT_VERIFY" == "true" ]]; then
    echo "Verifying artifats from '$DIST_DIR' directory"

    java -classpath ../../../../..//target/lib/commons-compress-1.4.1.jar:../../../../..//target/lib/commons-io-2.4.jar:. org/apache/sysml/validation/ValidateLicAndNotice $DIST_DIR

    cd "$ORIG_DIR" # Return to directoryt from it was called.
    exit 0
fi

cd "$ORIG_DIR" #return to original dir
echo "ERROR: wrong execution goals"
exit_with_usage
