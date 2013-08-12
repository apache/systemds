args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

X=readMM(paste(args[1], "X.mtx", sep=""))
Y=matrix(t(X),nrow=as.numeric(args[2]),ncol=as.numeric(args[3]),byrow=TRUE)
writeMM(as(Y,"CsparseMatrix"), paste(args[4], "Y", sep=""), format="text")




