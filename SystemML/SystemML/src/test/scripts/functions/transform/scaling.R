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

A = read.table(args[1], sep=",");
B = matrix(0, nrow=nrow(A), ncol=ncol(A));

cols = ncol(A);
A1 = A[, 1:cols/2];
A2 = A[,(cols/2+1):cols]
B[, 1:cols/2] = scale(A1, center=T, scale=F)
B[, (cols/2+1):cols] = scale(A2)

write.table(B, args[2], sep=",", row.names = F, col.names=F)
