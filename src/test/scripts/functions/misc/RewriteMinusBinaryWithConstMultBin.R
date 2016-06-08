args<-commandArgs(TRUE)
options(digits=22)
library("Matrix")

X=matrix(1,10,10)
Y=matrix(1,10,10)
lamda=7
S=X-lamda*Y
writeMM(as(S, "CsparseMatrix"), paste(args[2], "S", sep="")); 
