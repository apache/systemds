args <- commandArgs(TRUE)
library("Matrix")
library("moments")

m = as.numeric(args[1]);
n = as.numeric(args[2]);

load(file = "/local2/mboehm/parforIn.dat")
R <- array(0,dim=c(n,n))

for( i in 1:(n-1) )
{
   X <- V[ ,i];
   m2X <- moment(X,order=2);
   sigmaX <- sqrt(m2X * (m/(m-1.0)) );
                 
   for( j in (i+1):n )  
   {
      Y <- V[ ,j];  
      m2Y <- moment(Y,order=2);
      sigmaY <- sqrt(m2Y * (m/(m-1.0)) );
      covXY <- cov(X,Y);
      R[i,j] <- (covXY / (sigmaX*sigmaY));
   }
}   

save(R, file = "/local2/mboehm/parforOut.dat");
#writeMM(as(R, "CsparseMatrix"), "./tmpout/R");
