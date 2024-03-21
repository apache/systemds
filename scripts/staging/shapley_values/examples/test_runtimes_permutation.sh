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
#
#Runs systemds multiple times and stores resulting runtimes and sample sizes in file

data_file="../data/runtimes_permutation_test.csv"
permutations=10
samples=100

echo "Outputfile systemds: $data_file"

echo "instances,runtime_python,runtime_row,runtime_row_non_var,runtime_permutation,runtime_legacy,runtime_legacy_iterative" | tee "$data_file"
for instances in $(seq 0 250 2000); do
    #set to 1 on first run
    [[ $instances -eq 0 ]] && instances=1

    #take three samples per size
    for j in {1..3}; do
#        echo -n "${instances}," | tee -a "$data_file"
	
        #python
        runtime_python=$(python ./shap-permutation.py --n-permutations=${permutations} --n-instances=${instances} --silent --just-print-t)
        echo -n "${instances},${runtime_python}," | tee -a "$data_file"

        #by-row
        runtime_r=$(systemds ./shapley-multiLogReg-permutation-multirow.dml -stats 1 -nvargs n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-row 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
        echo -n "${runtime_r}," | tee -a "$data_file"

        #by-row non var
        runtime_r_non_var=$(systemds ./shapley-multiLogReg-permutation-multirow.dml -stats 1 -nvargs remove_non_var=1 n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-row 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
        echo -n "${runtime_r_non_var}," | tee -a "$data_file"

        #by-permutation
        runtime_p=$(systemds ./shapley-multiLogReg-permutation-multirow.dml -stats 1 -nvargs n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-permutation 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
        echo -n "${runtime_p}," | tee -a "$data_file"

        #legacy (only works for up to 2750 instances and crashes afterwards)
        runtime_l=$(systemds ./shapley-multiLogReg-permutation-multirow.dml -stats 1 -nvargs n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=legacy 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
        echo -n "${runtime_l}," | tee -a "$data_file"

        runtime_l_i=$(systemds ./shapley-multiLogReg-permutation-multirow.dml -stats 1 -nvargs n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=legacy-iterative 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
        echo -n "${runtime_l_i}" | tee -a "$data_file"

        #newline
	      echo "" | tee -a "$data_file"
    done
done
