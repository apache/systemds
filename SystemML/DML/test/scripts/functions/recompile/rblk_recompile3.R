args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))
V <- floor(V)
W <- table(as.vector(V))
writeMM(as(as.matrix(W), "CsparseMatrix"), paste(args[2], "Rout", sep="")); 