args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))
n <- ncol(V); 

R1 <- array(0,dim=c(1,n))
R2 <- array(0,dim=c(1,n))

for( i in 1:n )
{
   X <- V[ ,i];                 
   R1[1,i] <- sum(X);
}   

if( args[3]==1 )
{  
  for( i in 1:n )
  {
     X1 <- V[i,]; 
     X2 <- V[i,];                 
     R2[1,i] <- R1[1,i] + sum(X1)+sum(X2);
  }   
} else {
  for( i in 1:n )
  {
     X1 <- V[i,]; 
     X2 <- V[,i];                 
     R2[1,i] <- R1[1,i] + sum(X1)+sum(X2);
  }  
}

writeMM(as(R2, "CsparseMatrix"), paste(args[2], "Rout", sep="")); 