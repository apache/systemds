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


M = as.integer(args[1]);
N = as.integer(args[2]);

R = matrix(0, M, N);

for (x in 1 : M) {
    R[x,] = matrix (x, 1, N);
}

writeMM(as(R,"CsparseMatrix"), paste(args[3], "R", sep=""))

