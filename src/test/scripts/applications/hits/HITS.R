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

#library("batch")
# Usage:  R --vanilla --args graphfile G tol 1e-8 maxiter 100 < HITS.R
#parseCommandArgs()
# JUnit test class: dml.test.integration.applications.HITSTest.java
# command line invocation assuming $HITS_HOME is set to the home of the R script
# Rscript $HITS_HOME/HITSTest.R $HITS_HOME/in/ 2 0.000001 $HITS_HOME/expected/

args <- commandArgs(TRUE)
library("Matrix")


maxiter = as.integer(args[2]);
tol = as.double(args[3]);

G = readMM(paste(args[1], "G.mtx", sep=""));
authorities = round(G);
hubs = authorities

#N = nrow(G)
#D = ncol(G)

 
# HITS = power iterations to compute leading left/right singular vectors
 
#authorities = matrix(1.0/N,N,1)
#hubs = matrix(1.0/N,N,1)

converge = FALSE
iter = 0

while(!converge) {
	
	hubs_old = hubs
	hubs = G %*% authorities

	authorities_old = authorities
	authorities = t(G) %*% hubs

	hubs = hubs/max(hubs)
	authorities = authorities/max(authorities)

	delta_hubs = sum((hubs - hubs_old)^2)
	delta_authorities = sum((authorities - authorities_old)^2)

	converge = ((abs(delta_hubs) < tol) & (abs(delta_authorities) < tol) | (iter>maxiter))
	
	iter = iter + 1
	print(paste("Iterations :", iter, " delta_hubs :", delta_hubs, " delta_authorities :", delta_authorities))
}

writeMM(as(hubs,"CsparseMatrix"),paste(args[4], "hubs", sep=""));
writeMM(as(authorities,"CsparseMatrix"),paste(args[4], "authorities",sep=""));
