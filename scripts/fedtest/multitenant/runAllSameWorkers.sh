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

CMD=${1:-"systemds"}
DATADIR=${2:-"temp"}/sameworkers

export SYSDS_QUIET=1

FILENAME=$0
err_report() {
  echo "Error in $FILENAME on line $1"
}
trap 'err_report $LINENO' ERR

echo_stderr() {
  echo "$@" >&2
}

BASEPATH=$(dirname "$0")

# add SameWorkers scripts here
scripts=("runParforSumSameWorkers" "runSumSameWorkers" "runWSigmoidSameWorkers" "runALSSameWorkers")

globalErrorCount=0
for scriptName in "${scripts[@]}"
do
  echo "+++ Running ${scriptName}"
  trap "" ERR # disable error trapping
  ${BASEPATH}/${scriptName}.sh $CMD $DATADIR 1> /dev/null
  retVal=$? # get the return value of the previous command (failure count)
  trap 'err_report $LINENO' ERR # re-enable error trapping
  if (( $(echo "$retVal != 0" | bc -l) )); then
    echo_stderr "FAILURE: Encountered ${retVal} errors when executing ${scriptName}"
    globalErrorCount=$((globalErrorCount + retVal))
  else
    echo "SUCCESS: ${scriptName} was successful"
  fi
done

echo_stderr "ERRORS: ${globalErrorCount}"
exit $globalErrorCount # return the number of failures as exit value
