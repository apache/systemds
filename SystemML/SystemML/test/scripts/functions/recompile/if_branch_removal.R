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

X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))

if( as.integer(args[2])==1 )
{
   v = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v));
}

if( as.integer(args[2])!=1 )
{
   v = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v));
} else
{
   v1 = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v1));
   v2 = matrix(1,nrow(X),1);
   X = as.matrix(cbind(X, v2));
}

writeMM(as(X, "CsparseMatrix"), paste(args[3], "X", sep="")); 