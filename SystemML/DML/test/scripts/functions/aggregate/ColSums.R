args <- commandArgs(TRUE)

library("Matrix")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- t(colSums(A));

writeMM(as(B, "CsparseMatrix"), paste(args[2], "B", sep="")); 