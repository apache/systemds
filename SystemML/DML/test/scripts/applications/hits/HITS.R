#library("batch")
# Usage:  R --vanilla --args graphfile G tol 1e-8 maxiter 100 < HITS.R
#parseCommandArgs()
# JUnit test class: dml.test.integration.applications.HITSTest.java
# command line invocation assuming $HITS_HOME is set to the home of the R script
# Rscript $HITS_HOME/HITSTest.R $HITS_HOME/in/ 2 0.000001 $HITS_HOME/expected/

args <- commandArgs(TRUE)
library("Matrix")


maxiter = args[2];
tol = args[3];

G = readMM(paste(args[1], "graph.mtx", sep=""));
authorities = readMM(paste(args[1], "init_authorities.mtx", sep=""));
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

writeMM(as(hubs,"CsparseMatrix"),paste(args[3], "hubs", sep=""),format="text")
writeMM(as(authorities,"CsparseMatrix"),paste(args[3], "authorities",sep=""),format="text")
