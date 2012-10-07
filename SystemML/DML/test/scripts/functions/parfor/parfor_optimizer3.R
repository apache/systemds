args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))
n <- ncol(V); 
n2 <- n/2;

R <- array(0,dim=c(1,n2))

for( i in 1:n2 )
{
   X <- V[,i];                 
   Y <- V[,n-i+1];                
   R[1,i] <- sum(X)+sum(Y);
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep="")); 
