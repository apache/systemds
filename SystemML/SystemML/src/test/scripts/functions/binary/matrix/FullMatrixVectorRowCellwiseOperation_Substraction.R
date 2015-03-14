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

A1 <- readMM(paste(args[1], "A.mtx", sep=""))
A <- as.matrix(A1);
B1 <- readMM(paste(args[1], "B.mtx", sep=""))
B <- as.matrix(B1);


#C <- A-B; #not supported on row vectors
C <- t(t(A)-as.vector(t(B)))

writeMM(as(C, "CsparseMatrix"), paste(args[2], "C", sep="")); 


