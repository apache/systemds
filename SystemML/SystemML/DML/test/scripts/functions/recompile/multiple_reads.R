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

X <- readMM(paste(args[1], "X1.mtx", sep=""))

if( 1==1 )
{
   X <- readMM(paste(args[1], "X2.mtx", sep=""))
}

writeMM(as(X, "CsparseMatrix"), paste(args[2], "X", sep="")); 