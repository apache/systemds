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

A = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))

R = A;
R[3,7] = as.matrix(3);
R[3,8] = as.matrix(4);


writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 


