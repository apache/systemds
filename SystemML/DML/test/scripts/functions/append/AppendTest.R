args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")
A1=readMM(paste(args[1], "A.mtx", sep=""))
A = as.matrix(A1);
B1=readMM(paste(args[1], "B.mtx", sep=""))
B = as.matrix(B1);
C=cbind2(A, B)
writeMM(as(C,"CsparseMatrix"), paste(args[2], "C", sep=""), format="text")
