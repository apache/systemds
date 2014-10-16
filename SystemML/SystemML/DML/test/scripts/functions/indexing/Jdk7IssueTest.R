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

M = as.matrix(readMM(paste(args[1], "M.mtx", sep="")))

R1 = matrix(0, nrow(M), ncol(M));
for( i in 1:10 ) {
   R1[,i] = M[,i]; 
}

R = t(colSums(R1));
writeMM(as(R,"CsparseMatrix"), paste(args[2], "R", sep=""))