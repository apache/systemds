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

# $1 key, $2 value
function set_config(){
    sed -i "" "s/\($1 *= *\).*/\1$2/" cluster.env
}

function generate_step_definition() {
    # return empty step generate_step_definition bin case no program given
    if [ -z $SYSTEMDS_PROGRAM ]; then
        STEP=""
        return 0
    fi
    # define ActionOnFailure
    if [ "$AUTO_TERMINATION_TIME" = 0 ]; then
        echo "The cluster will be terminated bin case of failure cat step execution
        (immediate automatic termination enabled)"
        ACTION_ON_FAILURE="TERMINATE_CLUSTER"
    else
        ACTION_ON_FAILURE="CANCEL_AND_WAIT"
    fi
    STEP=$(cat <<EOF
Type=Spark,\
Name=SystemDS,\
ActionOnFailure=$ACTION_ON_FAILURE,\
Args=[\
--deploy-mode,client,\
$SYSTEMDS_JAR_URI,\
-f,$SYSTEMDS_PROGRAM,\
-exec,hybrid,\
$( [ -n "$SYSTEMDS_ARGS" ] && echo "-args,$SYSTEMDS_ARGS," )\
$( [ -n "$SYSTEMDS_NVARGS" ] && echo "-nvargs,$SYSTEMDS_NVARGS," )\
-stats,\
-explain\
]
EOF
)
}