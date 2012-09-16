# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $SS_HOME is set to the home of the R script
# Rscript $SS_HOME/ScaleScalePearsonRWithWeightsTest.R $SS_HOME/in/ $SS_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")
library("boot")
# Usage: R --vanilla -args Xfile X < ScaleScaleTest.R

#parseCommandArgs()
######################

X = readMM(paste(args[1], "X.mtx", sep=""))
Y = readMM(paste(args[1], "Y.mtx", sep=""))
WM = readMM(paste(args[1],"WM.mtx", sep=""))

# create a matrix from X and Y vectors
mat = cbind(X[,1], Y[,1]);

# corr is a function in "boot" package
R = corr(mat, WM[,1]);

write(R, paste(args[2], "PearsonR", sep=""));
