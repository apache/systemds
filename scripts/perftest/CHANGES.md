<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# New additions to the performance test suite
Most of the new files were copied from the deprecated performance test suite (scripts/perftestDeprecated) and refactored to call SystemDS with additional configuration.
Most of the new DML scripts were copied from scripts/algorithms to scripts/perftest/scripts and then adapted to use built-in functions, if available.

### General changes of perftest and the refactored files moved from perftestDeprecated to perftest
- Added line for intel oneapi MKL system variable initialization in the matrixmult script. The initialization is commented for now, as it would be executed by the runAll.sh.
- Added commented initialization for MKL system variables in the runAll.sh. 
- By default, shell scripts can now be invoked without any additional parameters, but optional arguments can be given for output folder and the command to be ran (MR, SPARK, ECHO).
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