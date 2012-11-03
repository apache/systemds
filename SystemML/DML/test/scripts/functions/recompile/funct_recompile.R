args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))
n <- ncol(V); 

R <- array(0,dim=c(n,1))

for( i in 1:n )
{
   X <- V[2:nrow(V),i];                 
   R[i,1] <- sum(X %*% t(X));
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep="")); 