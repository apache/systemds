# New additions to the performance test suite
Most of the new files were copied from the deprecated performance test suite (scripts/perftestDeprecated) and refactored to call SystemDS with additional configuration.
Most of the new DML scripts were copied from scripts/algorithms to scripts/perftest/scripts and then adapted to use built-in functions, if available.

### General changes of perftest and the refactored files moved from perftestDeprecated to perftest
- Removed the expected "<MR | SPARK | ECHO>" argument from copied scripts from the deprecated test suite.
- Added line for intel oneapi MKL system variable initialization in the matrixmult script.
- MKL system variables are now initialized once in the runAll.sh, not in subsequent microbenchmark scripts.
- By default, shell scripts can now be invoked without any additional parameters, but an optional folder path can be specified for outputs.
- Added SystemDS-config.xml in the perftest/conf folder, which is used by all refactored perftest scripts.
- Refactored all scripts so they must be ran from the SystemDS root, as described in the README.
- times.txt was moved to the "results" folder in perftest.
- Time measurements appended to results/times.txt are now additionally measured in microseconds instead of just seconds.
- All DML scripts, that are ultimately called by the microbenchmarks, can be found in perftest/scripts. This excludes the original algorithmic scripts that are still in use, if there was no corresponding built-in function.
- Removed the -explain flag from all systemds calls.


### Bash scripts that now call a new DML script that makes use of a built-in function, instead of a fully implemented algorithm
- perftest/runMultiLogReg.sh -> perftest/scripts/MultiLogReg.dml
- perftest/runMSVM.sh -> perftest/scripts/m-svm.dml
- perftest/runNaiveBayes.sh -> perftest/scripts/naive-bayes.dml
- perftest/runNaiveBayes.sh -> perftest/scripts/naive-bayes-predict.dml
- perftest/runMultiLogReg.sh -> perftest/scripts/MultiLogReg.dml
- perftest/runLinearRegCG.sh -> perftest/scripts/LinearRegCG.dml
- perftest/runLinearRegDS.sh -> perftest/scripts/LinearRegDS.dml
- perftest/runGLM_poisson_log.sh -> perftest/scripts/GLM.dml
- perftest/runGLM_gamma_log.sh -> perftest/scripts/GLM.dml
- perftest/runGLM_binomial_probit.sh -> perftest/scripts/GLM.dml


### Bash scripts still calling old DML scripts, which fully implement algorithms
- perftest/runMultiLogReg.sh -> algorithms/GLM-predict.dml
- perftest/runL2SVM.sh -> algorithms/l2-svm-predict.dml
- perftest/runMSVM.sh -> algorithms/m-svm-predict.dml
- perftest/runLinearRegCG.sh -> algorithms/GLM-predict.dml
- perftest/runLinearRegDS.sh -> algorithms/GLM-predict.dml
- perftest/runGLM_poisson_log.sh -> algorithms/GLM-predict.dml
- perftest/runGLM_gamma_log.sh -> algorithms/GLM-predict.dml
- perftest/runGLM_binomial_probit.sh -> algorithms/GLM-predict.dml

### Bash scripts that already did call a DML script with a single built-in functions (only needed some refactoring)
- perftest/runL2SVM.sh -> algorithms/l2-svm.dml (This already uses the built-in functopn l2svm!)


### Notes/Peculiarities
- Built-in MSVM is ~80% slower than the fully implemented MSVM DML script in /algorithms. I wonder why? (This was tested with MKL)
- perftest/scripts/naive-bayes-predict.dml only had a small part replaced with the built-in function. 

	
	
	
	
	
	

