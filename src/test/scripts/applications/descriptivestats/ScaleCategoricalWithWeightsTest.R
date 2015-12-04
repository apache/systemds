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

# JUnit test class: dml.test.integration.descriptivestats.BivariateScaleCategoricalTest.java
# command line invocation assuming $SC_HOME is set to the home of the R script
# Rscript $SC_HOME/ScaleCategorical.R $SC_HOME/in/ $SC_HOME/expected/

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")
# Usage: R --vanilla -args Xfile X < ScaleCategoricalTest.R

#parseCommandArgs()
######################
Atemp = readMM(paste(args[1], "A.mtx", sep=""));
Ytemp = readMM(paste(args[1], "Y.mtx", sep=""));
WM = readMM(paste(args[1], "WM.mtx", sep=""));

Yv=rep(Ytemp[,1],WM[,1])
Av=rep(Atemp[,1],WM[,1])

W = sum(WM);
my = sum(Yv)/W;
varY = var(Yv);

CFreqs = as.matrix(table(Av)); 
CMeans = as.matrix(aggregate(Yv, by=list(Av), "mean")$x);
CVars = as.matrix(aggregate(Yv, by=list(Av), "var")$x);

# number of categories
R = nrow(CFreqs);

Eta = sqrt(1 - ( sum((CFreqs-1)*CVars) / ((W-1)*varY) ));

anova_num = sum( (CFreqs*(CMeans-my)^2) )/(R-1);
anova_den = sum( (CFreqs-1)*CVars )/(W-R);
ANOVAF = anova_num/anova_den;

print(W, digits=15);
print(R, digits=15);
print(anova_num, digits=15);
print(anova_den, digits=15);

#######################

write(Eta, paste(args[2], "Eta", sep=""));

write(ANOVAF, paste(args[2], "AnovaF", sep=""));

write(varY, paste(args[2], "VarY", sep=""));

write(my, paste(args[2], "MeanY", sep=""));

writeMM(as(CVars,"CsparseMatrix"), paste(args[2], "CVars", sep=""), format="text");
writeMM(as(CFreqs,"CsparseMatrix"), paste(args[2], "CFreqs", sep=""), format="text");
writeMM(as(CMeans,"CsparseMatrix"), paste(args[2], "CMeans", sep=""), format="text");



