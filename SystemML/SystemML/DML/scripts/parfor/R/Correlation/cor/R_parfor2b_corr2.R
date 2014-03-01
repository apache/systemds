args <- commandArgs(TRUE)
library("Matrix")
library("doSNOW");

hosts <- c(
	rep("dml1", as.numeric(args[3])),
  rep("dml2", as.numeric(args[3])), 
	rep("dml3", as.numeric(args[3])), 
	rep("dml4", (as.numeric(args[3])-1)),
	rep("dml5", as.numeric(args[3]))
	)
cl <-makeCluster(hosts, type = "SOCK")
registerDoSNOW(cl)

m = as.numeric(args[1]);
n = as.numeric(args[2]);

load(file = "/local2/mboehm/parforIn.dat")
R <- array(0,dim=c(n,n))

Rtmp = (
foreach( i=1:(n-1), .combine=c ) %dopar%{
   tmp = c()
   X <- V[ ,i];                 
   for( j in (i+1):n )  
   {
      Y <- V[ ,j];  
      tmp = c(tmp, c(i,j, cor(X, Y) ))
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

stopCluster(cl)