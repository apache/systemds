#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < ScaleScaleTest.R

#parseCommandArgs()
######################

X = readMM(file="$$indir$$X.mtx")
Y = readMM(file="$$indir$$Y.mtx")
Helper=matrix(1, 2, 1)

# cor.test returns a list containing t-statistic, df, p-value, and R
cort = cor.test(X[,1], Y[,1]);

R = as.numeric(cort[4]);

RHelper = R * Helper;
writeMM(as(t(RHelper),"CsparseMatrix"), "$$Routdir$$outPearsonR", format="text");
