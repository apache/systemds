args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))
writeMM(as(V, "CsparseMatrix"), paste(args[2], "Rout", sep="")); 