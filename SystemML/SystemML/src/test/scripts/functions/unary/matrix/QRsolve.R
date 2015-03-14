
args <- commandArgs(TRUE)
library(Matrix)

# Rscript ./test/scripts/functions/unary/matrix/QRsolve.R ./test/scripts/functions/unary/matrix/in/A.mtx ./test/scripts/functions/unary/matrix/in/y.mtx ./test/scripts/functions/unary/matrix/expected/x

A = readMM(args[1]); #paste(args[1], "A.mtx", sep=""));
b = readMM(args[2]); #paste(args[1], "b.mtx", sep=""));

m = nrow(A);
n = ncol(A);

Ab = cbind(as.matrix(A), as.matrix(b));

Ab_qr = qr(Ab);
Rb = qr.R(Ab_qr); 

R = Rb[1:n, 1:n];
c = Rb[1:n, (n+1)];

x = solve(R,c);

writeMM(as(x, "CsparseMatrix"), args[3]); #paste(args[2], "x.mtx", sep=""));

