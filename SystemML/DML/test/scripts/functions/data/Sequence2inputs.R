args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

A = seq(as.numeric(args[1]), as.numeric(args[2]));
writeMM(as(A,"CsparseMatrix"), paste(args[3], "A", sep=""), format="text")

