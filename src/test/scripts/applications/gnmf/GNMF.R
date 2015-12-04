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

# JUnit test class: dml.test.integration.applications.GNMFTest.java
# command line invocation assuming $GNMF_HOME is set to the home of the R script
# Rscript $GNMF_HOME/GNMF.R $GNMF_HOME/in/ 3 $GNMF_HOME/expected/
args <- commandArgs(TRUE)
library(Matrix)

V = readMM(paste(args[1], "v.mtx", sep=""));
W = readMM(paste(args[1], "w.mtx", sep=""));
H = readMM(paste(args[1], "h.mtx", sep=""));
max_iteration = as.integer(args[2]);
i = 0;

Eps = 10^-8;

while(i < max_iteration) {
	H = H * ((t(W) %*% V) / (((t(W) %*% W) %*% H)+Eps)) ;
	W = W * ((V %*% t(H)) / ((W %*% (H %*% t(H)))+Eps));
	i = i + 1;
}

writeMM(as(W, "CsparseMatrix"), paste(args[3], "w", sep=""));
writeMM(as(H, "CsparseMatrix"), paste(args[3], "h", sep=""));