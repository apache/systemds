args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")));
c <- as.matrix(readMM(paste(args[1], "c.mtx", sep="")));

if( ncol(X)==1 )
{
   Y <- X[c];
} else {
   Y <- X[c,c];
}

writeMM(as(Y, "CsparseMatrix"), paste(args[2], "Y.mtx", sep="")); 