args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

V <- readMM(paste(args[1], "V.mtx", sep=""))

m <- nrow(V);
n <- ncol(V); 
W <- m;

R <- array(0,dim=c(n,n))

for( i in 1:8 )
{
   X <- V[ ,i];                 
      
   for( j in (i+1):(i+9) )  
   {
      Y <- V[ ,j];  
      R[i,j] <- cor(X, Y)  
      #print(R[i,j]);
   }
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep=""));
