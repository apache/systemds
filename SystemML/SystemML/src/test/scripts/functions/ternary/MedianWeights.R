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

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))

#without weights (assumes weights of 1)
s = median(A);
m = as.matrix(s);

writeMM(as(m, "CsparseMatrix"), paste(args[3], "R", sep="")); 


