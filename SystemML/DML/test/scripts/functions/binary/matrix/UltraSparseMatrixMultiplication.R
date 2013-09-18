args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")))

P <- diag( as.vector(B==2) )
Px <- P[rowSums((P==0) | is.na(P)) != ncol(P),];

C <- Px %*% A;

writeMM(as(C, "CsparseMatrix"), paste(args[2], "C", sep="")); 

