#library("batch")
library("Matrix")
library("boot")
# Usage: R --vanilla -args Xfile X < ScaleScaleTest.R

#parseCommandArgs()
######################

X = readMM(file="$$indir$$X.mtx")
Y = readMM(file="$$indir$$Y.mtx")
WM = readMM(file="$$indir$$WM.mtx")
Helper=matrix(1, 2, 1)

# create a matrix from X and Y vectors
mat = cbind(X[,1], Y[,1]);

# corr is a function in "boot" package
R = corr(mat, WM[,1]);

RHelper = R * Helper;
writeMM(as(t(RHelper),"CsparseMatrix"), "$$Routdir$$outPearsonR", format="text");
