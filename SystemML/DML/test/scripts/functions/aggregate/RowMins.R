args <- commandArgs(TRUE)

if(!("matrixStats" %in% rownames(installed.packages()))){
   install.packages("matrixStats")
}

library("Matrix")
library("matrixStats") 

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- rowMins(A);

writeMM(as(B, "CsparseMatrix"), paste(args[2], "B", sep="")); 