#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A <- readMM(paste(args[1], "A.mtx", sep=""))
col <- as.numeric(args[2]);

B <- A[order(sign(col)*A[,abs(col)]),]

writeMM(as(B, "CsparseMatrix"), paste(args[3], "B.mtx", sep="")); 