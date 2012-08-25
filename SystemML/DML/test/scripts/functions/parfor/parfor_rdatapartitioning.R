args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))
n <- nrow(V); 

R <- array(0,dim=c(1,n))

for( i in 1:n )
{
   X <- V[i,];                 
   R[1,i] <- sum(X);
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep=""));
