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

data_file="${1:-../data/runtimes_permutation_test.csv}"
permutations=3
samples=100


adult_data_sysds_str="data_dir=../data/adult/ X_bg_path=Adult_X.csv B_path=Adult_W.csv metadata_path=Adult_meta.csv model_type=multiLogRegPredict"
adult_data_python_str="--data-dir=../data/adult/ --data-x=Adult_X.csv --model-type=multiLogReg"

census_data_sysds_str="data_dir=../data/census/ X_bg_path=census_xTrain.csv B_path=census_bias.csv metadata_path=census_dummycoding_meta.csv model_type=l2svmPredict"
census_data_python_str="--data-dir=../data/census/ --data-x=census_xTrain.csv --data-y=census_yTrain_corrected.csv --model-type=l2svm"

exp_type_array=("adult_linlogreg" "census_l2svm")

echo "Outputfile: $data_file"

echo "exp_type,instances,runtime_python,runtime_row,runtime_row_non_var,runtime_row_partitioned,runtime_permutation,runtime_legacy,runtime_legacy_iterative" | tee "$data_file"
for instances in $(seq 0 500 20000); do
    #set to 1 on first run
    [[ $instances -eq 0 ]] && instances=1

    #take three samples per size
    for j in {1..3}; do
#        echo -n "${instances}," | tee -a "$data_file"
	      for exp_type in "${exp_type_array[@]}"; do
	        if [ "$exp_type" = "adult_linlogreg" ]; then
	            data_str=$adult_data_sysds_str
              py_str=$adult_data_python_str
          elif [ "$exp_type" = "census_l2svm" ]; then
              data_str=$census_data_sysds_str
              py_str=$census_data_python_str
          else
              echo "Exp type unknown: $exp_type"
              exit 1
          fi

          #python
          runtime_python=$(python ./shap-permutation.py ${py_str} --n-permutations=${permutations} --n-instances=${instances} --silent --just-print-t)
          echo -n "${exp_type},${instances},${runtime_python}," | tee -a "$data_file"

          #by-row
          runtime_r=$(systemds ./shapley-permutation-experiment.dml -stats 1 -nvargs ${data_str} n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-row 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
          echo -n "${runtime_r}," | tee -a "$data_file"

          #by-row non var
          runtime_r_non_var=$(systemds ./shapley-permutation-experiment.dml -stats 1 -nvargs ${data_str} remove_non_var=1 n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-row 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
          echo -n "${runtime_r_non_var}," | tee -a "$data_file"

          #by-row partitioned
          runtime_r_partitioned=$(systemds ./shapley-permutation-experiment.dml -stats 1 -nvargs ${data_str} remove_non_var=0 use_partitions=1 n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-row  2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
          echo -n "${runtime_r_partitioned}," | tee -a "$data_file"

          #by-permutation
          [[ $instances -le 10000 ]] && runtime_p=$(systemds ./shapley-permutation-experiment.dml -stats 1 -nvargs ${data_str} n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-permutation 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
          echo -n "${runtime_p}," | tee -a "$data_file"
          unset runtime_p

          #legacy (only works for up to 2750 instances and crashes afterwards)
          [[ $instances -le 2500 ]] && runtime_l=$(systemds ./shapley-permutation-experiment.dml -stats 1 -nvargs ${data_str} n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=legacy 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
          echo -n "${runtime_l}," | tee -a "$data_file"
          unset runtime_l

          [[ $instances -le 2500 ]] && runtime_l_i=$(systemds ./shapley-permutation-experiment.dml -stats 1 -nvargs ${data_str} n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=legacy-iterative 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
          echo -n "${runtime_l_i}" | tee -a "$data_file"
          unset runtime_l_i

          #newline
          echo "" | tee -a "$data_file"
        done
    done
done
