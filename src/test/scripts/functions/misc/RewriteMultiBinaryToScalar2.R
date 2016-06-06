X=matrix(1,10,10)
lamda=7
args<-commandArgs(TRUE)
write(sum(X*lamda),paste(args[2],"Scalar",sep=""))
