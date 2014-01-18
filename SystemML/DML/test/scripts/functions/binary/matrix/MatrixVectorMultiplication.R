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

A1 <- readMM(paste(args[1], "A.mtx", sep=""))
A <- as.matrix(A1);
x1 <- readMM(paste(args[1], "x.mtx", sep=""))
x <- as.matrix(x1);

y <- A%*%x;

writeMM(as(y, "CsparseMatrix"), paste(args[2], "y", sep="")); 


