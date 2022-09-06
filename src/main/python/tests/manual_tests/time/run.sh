# -------------------------------------------------------------
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
# -------------------------------------------------------------

export SYSDS_QUIET=1

tests=("startup_time_run for_loop_time_run")
tests=("for_loop_time_run")
base="tests/manual_tests/time/"
gr="elapsed"
gr="elapsed|task-clock|cycles|instructions"
rep=50

for t in $tests; do

    # Verbose runs. to verify it works.
    systemds $base$t.dml
    python $base$t.py

    # Timed runs
    # Systemds
    perf stat -d -d -d -r $rep \
        systemds $base$t.dml \
        2>&1 | grep -E $gr

    # PythonAPI SystemDS
    perf stat -d -d -d -r $rep \
        python $base$t.py \
        2>&1 | grep -E $gr
done
