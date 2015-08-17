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

S = matrix(1,100,1);
A = diag(as.vector(S));
B = A %*% S;
C = table (B, seq (1, nrow(A), 1));
R = colSums(C);
R = R %*% A;

writeMM(as(R, "CsparseMatrix"), paste(args[1], "R", sep="")); 