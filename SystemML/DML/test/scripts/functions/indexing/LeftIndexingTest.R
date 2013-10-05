#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

A1=readMM(paste(args[1], "A.mtx", sep=""))
B1=readMM(paste(args[1], "B.mtx", sep=""))
C1=readMM(paste(args[1], "C.mtx", sep=""))
D1=readMM(paste(args[1], "D.mtx", sep=""))
A=as.matrix(A1);
B=as.matrix(B1);
C=as.matrix(C1);
D=as.matrix(D1);

A[args[2]:args[3],args[4]:args[5]]=0
A[args[2]:args[3],args[4]:args[5]]=B
writeMM(as(A,"CsparseMatrix"), paste(args[6], "AB", sep=""), format="text")
A[1:args[3],args[4]:ncol(A)]=0
A[1:args[3],args[4]:ncol(A)]=C
writeMM(as(A,"CsparseMatrix"), paste(args[6], "AC", sep=""), format="text")
A[,args[4]:args[5]]=0
A[,args[4]:args[5]]=D
writeMM(as(A,"CsparseMatrix"), paste(args[6], "AD", sep=""), format="text")