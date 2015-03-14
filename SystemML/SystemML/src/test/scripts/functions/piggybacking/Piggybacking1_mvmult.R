
args <- commandArgs(TRUE)
library(Matrix)

A = readMM(args[1]);
x = readMM(args[2]);

y = A %*% x;
z = A %*% y;

writeMM(as(z, "CsparseMatrix"), args[3]); 

