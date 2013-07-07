args <- commandArgs(TRUE)
library("Matrix")

X = readMM("./mboehm/exp/in/x.mtx");
y = readMM("./mboehm/exp/in/y.mtx");
params = readMM("./mboehm/exp/in/params.mtx");
save(X, y, params, file = "/local2/mboehm/parforIn.dat");