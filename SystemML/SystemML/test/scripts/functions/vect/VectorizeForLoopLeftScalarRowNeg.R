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

A = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))

s = 0.0;
for( i in 1:nrow(A) ){
   ip = 1+i/2;
   s = s + A[ip,7];
}
R = A;
R[7,7] = s;

writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 


