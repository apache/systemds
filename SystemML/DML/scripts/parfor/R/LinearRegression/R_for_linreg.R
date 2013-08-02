args <- commandArgs(TRUE)
library("Matrix")

m = as.numeric(args[1]);
n = as.numeric(args[2]);
nc = as.numeric(args[3]);
k = as.numeric(args[4]);

load(file = "/local2/mboehm/parforIn.dat")
R = array(0,dim=c(k,nc))

DA = t(X) %*% X; # X'X
Db = t(X) %*% y; # X'y

for( i in 1:nc )
{
   c <- sort(sample(1:n, k, replace=F));
   A <- DA[c,c];
   b <- Db[c];
   beta <- solve(A, b, LINPACK=FALSE);
   R[,i] <- beta;
}   

save(R, file = "/local2/mboehm/parforOut.dat", compress=FALSE);
#writeMM(as(R, "CsparseMatrix"), "./tmpout/R");
