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

#!/bin/bash
#
#Runs systemds multiple times and stores resulting runtimes and sample sizes in file

#systemds shapley-multiLogReg.dml -stats 1 -nvargs samples_per_feature=200 2>/dev/null | grep "Total elapsed time" | awk '{print $4}'

multiplicator=5000
data_file="../data/systemds_runtimes.csv"

echo "Outputfile: $data_file"

echo "samples,runtime" | tee "$data_file"
for i in {1..8}; do
	samples=$((i*multiplicator))
	for j in {1..3}; do
		echo -n "${samples}," | tee "$data_file"
		runtime=$(systemds shapley-multiLogReg.dml -stats 1 -nvargs "samples_per_feature=$samples" 2>/dev/null | grep "Total elapsed time" | awk '{print $4}' | tr \, \.)
		echo "$runtime" | tee "$data_file"
	done
done
