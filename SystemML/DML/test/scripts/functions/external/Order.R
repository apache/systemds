args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A <- readMM(paste(args[1], "A.mtx", sep=""))
col <- as.numeric(args[2]);

B <- A[order(sign(col)*A[,abs(col)]),]

writeMM(as(B, "CsparseMatrix"), paste(args[3], "B.mtx", sep="")); 