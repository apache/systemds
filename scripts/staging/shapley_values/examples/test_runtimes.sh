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
