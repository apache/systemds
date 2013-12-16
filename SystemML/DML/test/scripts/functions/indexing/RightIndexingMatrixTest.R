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

A1=readMM(paste(args[1], "A.mtx", sep=""))
A = as.matrix(A1);

B=A[args[2]:args[3],args[4]:args[5]]
C=A[1:args[3],args[4]:ncol(A)]
D=A[,args[4]:args[5]]
writeMM(as(B,"CsparseMatrix"), paste(args[6], "B", sep=""), format="text")
writeMM(as(C,"CsparseMatrix"), paste(args[6], "C", sep=""), format="text")
writeMM(as(D,"CsparseMatrix"), paste(args[6], "D", sep=""), format="text")
