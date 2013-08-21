args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")
A1=readMM(paste(args[1], "A.mtx", sep=""))
A = as.vector(A1);
B=diag(A)
C=matrix(1, nrow(B), ncol(B));
D=B%*%C
C=B+D
writeMM(as(C,"CsparseMatrix"), paste(args[2], "C", sep=""), format="text")

