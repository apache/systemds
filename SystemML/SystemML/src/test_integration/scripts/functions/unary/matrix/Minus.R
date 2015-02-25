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

X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))

Y = -X;

writeMM(as(Y, "CsparseMatrix"), paste(args[2], "Y", sep="")); 


