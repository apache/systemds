#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2015
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""))
v = readMM(paste(args[1], "v.mtx", sep=""))
maxiter = as.double(args[2]);

i = 0;
while(i < maxiter) {
	v = t(X) %*% (X %*% v);
	i = i + 1;
}

writeMM(as(v,"CsparseMatrix"), paste(args[3], "w", sep=""))
