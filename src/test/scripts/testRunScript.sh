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

# directory to write test case temporary files
TEMP=~/temp/systemml_test

# should this test suite continue after encountering an error? true false
CONTINUE_ON_ERROR=true

# expecting this test script to be in directory ${PROJECT_ROOT_DIR}/src/test/scripts
TEST_SCRIPT_REL_DIR=src/test/scripts

# expecting the run script to be in directory ${PROJECT_ROOT_DIR}/bin
RUN_SCRIPT=systemml

# the DML script with arguments we use to test the run script
DML_SCRIPT=genLinearRegressionData.dml
DML_SCRIPT_PATH=scripts/datagen
DML_OUTPUT=linRegData.csv
DML_ARGS="-nvargs numSamples=1000 numFeatures=50 maxFeatureValue=5 maxWeight=5 addNoise=FALSE b=0 sparsity=0.7 output=${DML_OUTPUT} format=csv perc=0.5"
DML_SCRIPT_WITH_ARGS="${DML_SCRIPT} ${DML_ARGS}"

# try to find the project root directory
USER_DIR="$PWD"
TEST_SCRIPT_PATH=$( dirname "$0" )
PROJECT_ROOT_DIR=$( cd "${TEST_SCRIPT_PATH/\/$TEST_SCRIPT_REL_DIR/}" ; pwd -P )

# generate the test log file name
DATE_TIME="$(date +"%Y-%m-%d-%H-%M-%S")"
TEST_LOG="${PROJECT_ROOT_DIR}/temp/test_runScript_${DATE_TIME}.log"


# verify we found the ${PROJECT_ROOT_DIR} and ${RUN_SCRIPT}
if [ "${TEST_SCRIPT_PATH}" = "${PROJECT_ROOT_DIR}" ]
then
    echo This test script "$0" is expected to be located in folder "${TEST_SCRIPT_REL_DIR}" under the project root.
    echo Please update "$0" and correctly set the variable "TEST_SCRIPT_REL_DIR".
    exit 1
fi
if [ ! -f "${PROJECT_ROOT_DIR}/bin/${RUN_SCRIPT}" ]
then
    echo "Could not find \"bin/${RUN_SCRIPT}\" in the project root directory. If the actual path of the run script is not \"bin/${RUN_SCRIPT}\", or if the actual project root directory is not \"${PROJECT_ROOT_DIR}\", please update this test script \"$0\"."
    exit 1
fi


# set an exit trap if we should exit on first error
if [ "${CONTINUE_ON_ERROR}" = "false" ]
then
    trap "exit 1" TERM
    export PID=$$
fi


# test setup, create temp folders
if [ ! -d "${TEMP}" ] ; then
    mkdir -p "${TEMP}"
fi
if [ ! -d "${PROJECT_ROOT_DIR}/temp" ] ; then
    mkdir -p "${PROJECT_ROOT_DIR}/temp"
fi


# start running the test cases and write output to log file
echo "Writing test log to file \"${TEST_LOG}\"."; echo "$( date +"%F %T" )" > ${TEST_LOG}


# invoke the run script from the project root directory
CURRENT_TEST="Test_root__DML_script_with_path"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"     >> ${TEST_LOG}
    cd "${PROJECT_ROOT_DIR}"
    echo "Working directory: $PWD"                  >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                           >  /dev/null   2>&1
    rm -f "temp/${DML_OUTPUT}"                      >  /dev/null   2>&1
    CMD="./bin/${RUN_SCRIPT} ${DML_SCRIPT_PATH}/${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                   >> ${TEST_LOG} 2>&1
    eval ${CMD}                                     >> ${TEST_LOG} 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]
    then
        if [    -f "${DML_OUTPUT}"      ] ; then ERR_MSG="${ERR_MSG}, outputdata in project root" ; EXIT_CODE=1 ; fi
        if [ !  -f "temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;      EXIT_CODE=1 ; fi
    fi
    if [ $EXIT_CODE -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)
CURRENT_TEST="Test_root__DML_script_file_name"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"      >> ${TEST_LOG}
    cd "${PROJECT_ROOT_DIR}"
    echo "Working directory: $PWD"                   >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                            >  /dev/null   2>&1
    rm -f "temp/${DML_OUTPUT}"                       >  /dev/null   2>&1
    CMD="./bin/${RUN_SCRIPT} ${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                    >> ${TEST_LOG} 2>&1
    eval ${CMD}                                      >> ${TEST_LOG} 2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [   -f "${DML_OUTPUT}"      ] ; then ERR_MSG="${ERR_MSG}, outputdata in project root" ; EXIT_CODE=1 ; fi
        if [ ! -f "temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;      EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)

# invoke the run script from the bin directory
CURRENT_TEST="Test_bin__DML_script_with_path"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${PROJECT_ROOT_DIR}/bin"
    echo "Working directory: $PWD"                >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null    2>&1
    rm -f "../temp/${DML_OUTPUT}"                 >  /dev/null    2>&1
    CMD="./${RUN_SCRIPT} ../${DML_SCRIPT_PATH}/${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                 >> ${TEST_LOG}  2>&1
    sh ${CMD}                                     >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [   -f "${DML_OUTPUT}"         ] ; then ERR_MSG="${ERR_MSG}, outputdata in bin folder" ; EXIT_CODE=1 ; fi
        if [ ! -f "../temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;    EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)
CURRENT_TEST="Test_bin__DML_script_file_name"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"  >> ${TEST_LOG}
    cd "${PROJECT_ROOT_DIR}/bin"
    echo "Working directory: $PWD"               >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                        >  /dev/null    2>&1
    rm -f "../temp/${DML_OUTPUT}"                >  /dev/null    2>&1
    CMD="./${RUN_SCRIPT} ${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                >> ${TEST_LOG}  2>&1
    sh ${CMD}                                    >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [   -f "${DML_OUTPUT}"         ] ; then ERR_MSG="${ERR_MSG}, outputdata in bin folder" ; EXIT_CODE=1 ; fi
        if [ ! -f "../temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;    EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)

# invoke the run script from a working directory outside of the project root
CURRENT_TEST="Test_out__DML_script_with_path"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n" >> ${TEST_LOG}
    cd "${TEMP}"
    echo "Working directory: $PWD"              >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                       >  /dev/null    2>&1
    CMD="\"${PROJECT_ROOT_DIR}/bin/${RUN_SCRIPT}\" \"${PROJECT_ROOT_DIR}/${DML_SCRIPT_PATH}/${DML_SCRIPT}\" ${DML_ARGS}"
    echo "${CMD}"                               >> ${TEST_LOG}  2>&1
    eval  ${CMD}                                >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [ ! -f "${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ; EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)
CURRENT_TEST="Test_out__DML_script_file_name"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${TEMP}"
    echo "Working directory: $PWD"                >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null   2>&1
    CMD="\"${PROJECT_ROOT_DIR}/bin/${RUN_SCRIPT}\" ${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                 >> ${TEST_LOG}  2>&1
    eval ${CMD}                                   >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [ ! -f "${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ; EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)


# test again from a directory with spaces in its path name
echo "Running test cases again from a folder with spaces in path name..."

SPACE_DIR="${TEMP}/Space Folder/SystemML"
if [ ! -d "${SPACE_DIR}" ]
then
    echo "mkdir -p ${SPACE_DIR}"    >> ${TEST_LOG}
    mkdir -p "${SPACE_DIR}"         >> ${TEST_LOG}
fi

printf "Copying project contents from \"${PROJECT_ROOT_DIR}\" to \"${SPACE_DIR}\"... "  >> ${TEST_LOG}
rsync -aqz --exclude '*.java' \
           --exclude 'Test*.*' \
           --exclude 'test-classes/' \
           --exclude 'hadoop-test/' \
           --exclude 'src/test' \
           --exclude 'docs/' \
           --exclude '.git' \
           --exclude '.idea' \
           --exclude '.settings' \
           "${PROJECT_ROOT_DIR}/" "${SPACE_DIR}"  >> ${TEST_LOG}
printf "done.\n"                                  >> ${TEST_LOG}


# invoke the run script from the project root directory
CURRENT_TEST="Test_space_root__DML_script_with_path"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${SPACE_DIR}"
    echo "Working directory: \"$PWD\""            >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null   2>&1
    rm -f "temp/${DML_OUTPUT}"                    >  /dev/null   2>&1
    CMD="./bin/${RUN_SCRIPT} ${DML_SCRIPT_PATH}/${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                 >> ${TEST_LOG} 2>&1
    sh ${CMD}                                     >> ${TEST_LOG} 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]
    then
        if [    -f "${DML_OUTPUT}"      ] ; then ERR_MSG="${ERR_MSG}, outputdata in project root" ; EXIT_CODE=1 ; fi
        if [ !  -f "temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;      EXIT_CODE=1 ; fi
    fi
    if [ $EXIT_CODE -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)
CURRENT_TEST="Test_space_root__DML_script_file_name"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${SPACE_DIR}"
    echo "Working directory: \"$PWD\""            >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null   2>&1
    rm -f "temp/${DML_OUTPUT}"                    >  /dev/null   2>&1
    CMD="./bin/${RUN_SCRIPT} ${DML_SCRIPT_WITH_ARGS} "
    echo "${CMD}"                                 >> ${TEST_LOG} 2>&1
    sh ${CMD}                                     >> ${TEST_LOG} 2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [   -f "${DML_OUTPUT}"      ] ; then ERR_MSG="${ERR_MSG}, outputdata in project root" ; EXIT_CODE=1 ; fi
        if [ ! -f "temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;      EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)

# invoke the run script from the bin directory
CURRENT_TEST="Test_space_bin__DML_script_with_path"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${SPACE_DIR}/bin"
    echo "Working directory: \"$PWD\""            >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null    2>&1
    rm -f "../temp/${DML_OUTPUT}"                 >  /dev/null    2>&1
    CMD="./${RUN_SCRIPT} ../${DML_SCRIPT_PATH}/${DML_SCRIPT_WITH_ARGS} "
    echo "${CMD}"                                 >> ${TEST_LOG}  2>&1
    sh ${CMD}                                     >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [   -f "${DML_OUTPUT}"         ] ; then ERR_MSG="${ERR_MSG}, outputdata in bin folder" ; EXIT_CODE=1 ; fi
        if [ ! -f "../temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;    EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)
CURRENT_TEST="Test_space_bin__DML_script_file_name"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${SPACE_DIR}/bin"
    echo "Working directory: \"$PWD\""            >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null    2>&1
    rm -f "../temp/${DML_OUTPUT}"                 >  /dev/null    2>&1
    CMD="./${RUN_SCRIPT} ${DML_SCRIPT_WITH_ARGS} "
    echo "${CMD}"                                 >> ${TEST_LOG}  2>&1
    sh ${CMD}                                     >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [   -f "${DML_OUTPUT}"         ] ; then ERR_MSG="${ERR_MSG}, outputdata in bin folder" ; EXIT_CODE=1 ; fi
        if [ ! -f "../temp/${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ;    EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)

# invoke the run script from a working directory outside of the project root
CURRENT_TEST="Test_space_out__DML_script_with_path"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${TEMP}"
    echo "Working directory: \"$PWD\""            >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null    2>&1
    CMD="\"${SPACE_DIR}/bin/${RUN_SCRIPT}\" \"${SPACE_DIR}/${DML_SCRIPT_PATH}/${DML_SCRIPT}\" ${DML_ARGS}"
    echo "${CMD}"                                 >> ${TEST_LOG}  2>&1
    eval ${CMD}                                   >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [ ! -f "${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ; EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)
CURRENT_TEST="Test_space_out__DML_script_file_name"
(
    printf "Running test \"${CURRENT_TEST}\"... "
    printf "\nTest case: \"${CURRENT_TEST}\"\n"   >> ${TEST_LOG}
    cd "${TEMP}"
    echo "Working directory: \"$PWD\""            >> ${TEST_LOG}
    rm -f "${DML_OUTPUT}"                         >  /dev/null    2>&1
    CMD="\"${SPACE_DIR}/bin/${RUN_SCRIPT}\" ${DML_SCRIPT_WITH_ARGS}"
    echo "${CMD}"                                 >> ${TEST_LOG}  2>&1
    eval ${CMD}                                   >> ${TEST_LOG}  2>&1
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]
    then
        if [ ! -f "${DML_OUTPUT}" ] ; then ERR_MSG="${ERR_MSG}, outputdata is missing" ; EXIT_CODE=1 ; fi
    fi
    if [ ${EXIT_CODE} -eq 0 ] ;
        then printf "success\n";
        else printf "failed${ERR_MSG}\n"; if [ "${CONTINUE_ON_ERROR}" = "false" ] ; then kill -s TERM $PID ; fi
    fi
)





# Cleanup

printf "Cleaning up temporary files and folders ..."
if [ -d "${TEMP}" ] ; then
  rm -rf "${TEMP}"
fi
printf "done.\n"

echo "Test log was written to file ${TEST_LOG}."

cd "${USER_DIR}"
