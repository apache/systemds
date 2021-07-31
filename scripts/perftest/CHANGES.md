# New additions to the performance test suite
Most of the new files were copied from the deprecated performance test suite (scripts/perftestDeprecated) and refactored to call SystemDS with additional configuration.
Most of the new DML scripts were copied from scripts/algorithms to scripts/perftest/scripts and then adapted to use built-in functions, if available.

### General changes of perftest and the refactored files moved from perftestDeprecated to perftest
- Added line for intel oneapi MKL system variable initialization in the matrixmult script. The initialization is commented for now, as it would be executed by the runAll.sh.
- Added commented initialization for MKL system variables in the runAll.sh. 
- By default, shell scripts can now be invoked without any additional parameters, but an optional folder path can be specified for outputs or for the command to be ultimately ran (local, spark, "debug").
- Added SystemDS-config.xml in the perftest/conf folder, which is used by all refactored perftest scripts.
- times.txt was moved to the "results" folder in perftest.
- Time measurements appended to results/times.txt are now additionally measured in microseconds instead of just seconds, for the smaller data benchmarks.
- All DML scripts, that are ultimately called by the microbenchmarks, can be found in perftest/scripts. This excludes the original algorithmic scripts that are still in use, if there was no corresponding built-in function.
- Removed the -explain flag from all systemds calls.

### Bash scripts that now call a new DML script that makes use of a built-in function, instead of a fully implemented algorithm
- perftest/runMultiLogReg.sh -> perftest/scripts/MultiLogReg.dml
- perftest/runL2SVM.sh -> perftest/scripts/l2-svm-predict.dml
- perftest/runMSVM.sh -> perftest/scripts/m-svm.dml
- perftest/runMSVM.sh -> perftest/scripts/m-svm-predict.dml
- perftest/runNaiveBayes.sh -> perftest/scripts/naive-bayes.dml
- perftest/runNaiveBayes.sh -> perftest/scripts/naive-bayes-predict.dml
- perftest/runLinearRegCG.sh -> perftest/scripts/LinearRegCG.dml
- perftest/runLinearRegDS.sh -> perftest/scripts/LinearRegDS.dml
- perftest/runGLM_poisson_log.sh -> perftest/scripts/GLM.dml
- perftest/runGLM_gamma_log.sh -> perftest/scripts/GLM.dml
- perftest/runGLM_binomial_probit.sh -> perftest/scripts/GLM.dml


### Bash scripts still calling old DML scripts, which fully implement algorithms
- perftest/runMultiLogReg.sh -> algorithms/GLM-predict.dml
- perftest/runLinearRegCG.sh -> algorithms/GLM-predict.dml
- perftest/runLinearRegDS.sh -> algorithms/GLM-predict.dml
- perftest/runGLM_poisson_log.sh -> algorithms/GLM-predict.dml
- perftest/runGLM_gamma_log.sh -> algorithms/GLM-predict.dml
- perftest/runGLM_binomial_probit.sh -> algorithms/GLM-predict.dml

### Bash scripts that already did call a DML script with a single built-in functions (only needed some refactoring)
- perftest/runL2SVM.sh -> algorithms/l2-svm.dml (This already uses the built-in function l2svm!)



	
	
	
	
	
	

