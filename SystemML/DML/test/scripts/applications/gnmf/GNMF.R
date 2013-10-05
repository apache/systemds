#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
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