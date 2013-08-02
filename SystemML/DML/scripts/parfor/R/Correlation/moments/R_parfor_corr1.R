args <- commandArgs(TRUE)
library("Matrix")
library("doMC");

registerDoMC(as.numeric(args[3]));

m = as.numeric(args[1]);
n = as.numeric(args[2]);

load(file = "/local2/mboehm/parforIn.dat")
R <- array(0,dim=c(n,n))

Rtmp = (
foreach( i=1:(n-1), .combine=c ) %dopar%{
   library("moments") #suppressMessages
   tmp = c()
   X <- V[ ,i];
   m2X <- moment(X,order=2);
   sigmaX <- sqrt(m2X * (m/(m-1.0)) );
                 
   for( j in (i+1):n )  
   {
      Y <- V[ ,j];  
      m2Y <- moment(Y,order=2);
      sigmaY <- sqrt(m2Y * (m/(m-1.0)) );
      covXY <- cov(X,Y);
      tmp = c(tmp, c(i,j,(covXY / (sigmaX*sigmaY))))
   }
   return (tmp);
}   
)

for( i in 1:length(Rtmp) )
{
  X = Rtmp[i];
  R[ X[1], X[2] ] = X[3];
}

save(R, file = "/local2/mboehm/parforOut.dat", compress=FALSE);
#writeMM(as(R, "CsparseMatrix"), "./tmpout/R");
