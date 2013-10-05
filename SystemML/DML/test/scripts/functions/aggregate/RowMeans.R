#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)

library("Matrix")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- rowMeans(A);

writeMM(as(B, "CsparseMatrix"), paste(args[2], "B", sep="")); 