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

A = readMM(paste(args[1], "A.mtx", sep=""))
by = as.integer(args[2]);
desc = (sum(A)>100)
ixret = (sum(A)>200)
col = A[,by];

if( ixret ) {
  B = order(col, decreasing=desc);
} else {
  B = A[order(col, decreasing=desc),];
}

writeMM(as(B,"CsparseMatrix"), paste(args[5], "B", sep=""))




