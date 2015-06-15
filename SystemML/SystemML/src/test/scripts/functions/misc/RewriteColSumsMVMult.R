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

X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
R = colSums( X * seq(1,nrow(X))%*%matrix(1,1,ncol(X)) );

writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 
