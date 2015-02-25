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

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")))

C = t(A) %*% B;

writeMM(as(C, "CsparseMatrix"), paste(args[2], "C", sep="")); 


