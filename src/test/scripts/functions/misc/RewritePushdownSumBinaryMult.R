X=matrix(1,10,10)
lamda=sum(X)
args<-commandArgs(TRUE)
write(sum(lamda*X),paste(args[2],"Scalar",sep=""))
