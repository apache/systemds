args <- commandArgs(TRUE)
library("Matrix")
library("doSNOW");

hosts <- c(
	rep("dml1", as.numeric(args[5])),
  rep("dml2", as.numeric(args[5])), 
	rep("dml3", as.numeric(args[5])), 
	rep("dml4", as.numeric(args[5])),
	rep("dml5", as.numeric(args[5]))
	)
cl <- makeCluster(hosts, type = "SOCK")
registerDoSNOW(cl)

m = as.numeric(args[1]);
n = as.numeric(args[2]);
nc = as.numeric(args[3]);
k = as.numeric(args[4]);

load(file = "/local2/mboehm/parforIn.dat")

DA = t(X) %*% X; # X'X
Db = t(X) %*% y; # X'y

R = (
foreach( i=1:nc, .combine=cbind, .multicombine=TRUE, .maxcombine=500 ) %dopar%{
   c <- sort(sample(1:n, k, replace=F));
   A <- DA[c,c];
   b <- Db[c];
   beta <- solve(A, b, LINPACK=FALSE);

   return (beta);
}   
)

save(R, file = "/local2/mboehm/parforOut.dat", compress=FALSE);
#writeMM(as(R, "CsparseMatrix"), "./tmpout/R");

stopCluster(cl)

 