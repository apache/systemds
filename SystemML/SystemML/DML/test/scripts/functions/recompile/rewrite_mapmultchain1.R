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

X <- readMM(paste(args[1], "X.mtx", sep=""))
P <- readMM(paste(args[1], "P.mtx", sep=""))
v <- readMM(paste(args[1], "v.mtx", sep=""))
k = ncol(P);

Q = P * (X %*% v);
HV = t(X) %*% (Q - P * (rowSums (Q) %*% matrix(1, 1, k)));

writeMM(as(HV, "CsparseMatrix"), paste(args[2], "HV", sep="")); 