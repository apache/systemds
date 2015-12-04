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

# JUnit test class: dml.test.integration.descriptivestats.BivariateOrdinalOrdinalWithWeightsTest.java
# command line invocation assuming $OO_HOME is set to the home of the R script
# Rscript $OO_HOME/OrdinalOrdinal.R $OO_HOME/in/ $OO_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

Atemp  = readMM(paste(args[1], "A.mtx", sep=""))
Btemp  = readMM(paste(args[1], "B.mtx", sep=""))
WMtemp = readMM(paste(args[1], "WM.mtx", sep=""))

#Atemp  = readMM(file="$$indir$$A.mtx"); #readMM(paste(args[1], "A.mtx", sep=""))
#Btemp  = readMM(file="$$indir$$B.mtx"); #readMM(paste(args[1], "B.mtx", sep=""))
#WMtemp = readMM(file="$$indir$$WM.mtx"); #readMM(paste(args[1], "WM.mtx", sep=""))

A = rep(Atemp[,1],WMtemp[,1])
B = rep(Btemp[,1],WMtemp[,1])

spearman = cor(A, B, method="spearman");

#paste("Weighted R value", spearman);

write(spearman, paste(args[2], "Spearman", sep=""));

