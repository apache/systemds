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

# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $CC_HOME is set to the home of the R script
# Rscript $CC_HOME/CategoricalCategoricalWithWeightsTest.R $CC_HOME/in/ $CC_HOME/expected/
# Usage: R --vanilla -args Xfile X < CategoricalCategoricalWithWeightsTest.R

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

#parseCommandArgs()
######################

print(commandArgs(TRUE)[1])

A = readMM(paste(args[1], "A.mtx", sep=""));
B = readMM(paste(args[1], "B.mtx", sep=""));
WM = readMM(paste(args[1], "WM.mtx", sep=""));

Av = A[,1];
Bv = B[,1];
WMv = WM[,1];

# create a data frame with vectors A, B, WM
df = data.frame(Av,Bv,WMv);

# contingency table with weights
F = xtabs ( WMv ~ Av + Bv, df);

# chisq.test returns a list containing statistic, p-value, etc.
cst = chisq.test(F);

# get the chi-squared coefficient from the list
chi_squared = as.numeric(cst[1]);
pValue = as.numeric(cst[3]);

write(pValue, paste(args[2], "PValue", sep=""));

#######################

q = min(dim(F));
W = sum(F);
cramers_v = sqrt(chi_squared/(W*(q-1)));

write(cramers_v, paste(args[2], "CramersV", sep=""));

