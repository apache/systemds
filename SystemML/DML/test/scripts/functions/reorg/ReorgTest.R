args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")
A1=readMM(paste(args[1], "A.mtx", sep=""))
A = as.vector(A1);
B=diag(A)
C=B+7
writeMM(as(C,"CsparseMatrix"), paste(args[2], "C", sep=""), format="text")

