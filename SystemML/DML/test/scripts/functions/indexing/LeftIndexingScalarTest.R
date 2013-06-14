args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

A1=readMM(paste(args[1], "A.mtx", sep=""))
A=as.matrix(A1);

A[13:13,1026:1026] = 7;
A[14:14,1027:1027] = 7*7;
A[1013,26] = 7;
A[1014,27] = 7*7;

writeMM(as(A,"CsparseMatrix"), paste(args[2], "A", sep=""), format="text")