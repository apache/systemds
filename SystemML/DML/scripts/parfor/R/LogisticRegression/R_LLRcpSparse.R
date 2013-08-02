args <- commandArgs(TRUE)
library("Matrix")

X = readMM("/local2/mboehm/llr/x.mtx");
y = readMM("/local2/mboehm/llr/y.mtx");
params = readMM("/local2/mboehm/llr/params.mtx");
save(X, y, params, file = "/local2/mboehm/parforIn.dat", compress=FALSE);