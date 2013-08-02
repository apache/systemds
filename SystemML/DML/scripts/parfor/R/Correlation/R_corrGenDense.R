args <- commandArgs(TRUE)

library("Matrix")

m = as.numeric(args[1]);
n = as.numeric(args[2]);

V <- array(runif(m*n, min=0, max=1),dim=c(m,n))
save(V, file = "/local2/mboehm/parforIn.dat", compress=FALSE)
