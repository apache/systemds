args <- commandArgs(TRUE)

library("Matrix")

m = as.numeric(args[1]);
n = as.numeric(args[2]);

X <- array(runif(m*n, min=0, max=1),dim=c(m,n))
y = array(runif(m, min=0, max=1),dim=c(m,1))
save(X, y, file = "/local2/mboehm/parforIn.dat")