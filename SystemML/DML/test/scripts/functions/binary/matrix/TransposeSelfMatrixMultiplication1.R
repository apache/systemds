args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A1 <- readMM(paste(args[1], "A.mtx", sep=""))
A <- as.matrix(A1);

B <- t(A)%*%A;

writeMM(as(B, "CsparseMatrix"), paste(args[2], "B", sep="")); 