#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2014
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))

Y = as.matrix(aggregate(X[,1] ~ X[,2], data=X, length)[,1]);
Z = matrix(7, nrow(Y)+ncol(Y), nrow(Y)+ncol(Y)+1);

writeMM(as(Z, "CsparseMatrix"), paste(args[2], "Z", sep="")); 