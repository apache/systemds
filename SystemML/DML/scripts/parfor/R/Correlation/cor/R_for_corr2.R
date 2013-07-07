args <- commandArgs(TRUE)
library("Matrix")

m <- as.numeric(args[1]);
n <- as.numeric(args[2]);

load(file = "/local2/mboehm/parforIn.dat")
R <- array(0,dim=c(n,n))

for( i in 1:(n-1) )
{
   X <- V[ ,i]; 
   for( j in (i+1):n )  
   {
      Y <- V[ ,j];
      R[i,j] <- cor(X, Y)  
   }
}   

save(R, file = "/local2/mboehm/parforOut.dat");
#writeMM(as(R, "CsparseMatrix"), "./tmpout/R");