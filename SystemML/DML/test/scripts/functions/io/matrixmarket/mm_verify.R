
args <- commandArgs(TRUE)
options(digits=22)

library(Matrix);

A = readMM(args[1]);
x =  sum(A);
write(x, args[2]);

