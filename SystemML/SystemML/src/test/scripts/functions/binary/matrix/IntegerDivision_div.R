#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2014
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
#options(digits=22)

library("Matrix")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")))
if( nrow(A)==1 ){ #support for scalars        
   A <- as.numeric(A);
}
if( nrow(B)==1 ){ #support for scalars
   B <- as.numeric(B);
}
C <- A%/%B;

#note: writeMM replaces NaN and Inf
writeMM(as(C, "CsparseMatrix"), paste(args[2], "C", sep="")); 


