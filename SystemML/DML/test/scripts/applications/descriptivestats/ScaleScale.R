# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $SS_HOME is set to the home of the R script
# Rscript $SS_HOME/ScaleScale.R $SS_HOME/in/ $SS_HOME/expected/
args <- commandArgs(TRUE)

library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""))
Y = readMM(paste(args[1], "Y.mtx", sep=""))
Helper=matrix(1, 2, 1)

# cor.test returns a list containing t-statistic, df, p-value, and R
cort = cor.test(X[,1], Y[,1]);

R = as.numeric(cort[4]);

RHelper = R * Helper;
writeMM(as(t(RHelper),"CsparseMatrix"), paste(args[2], "PearsonR", sep=""), format="text");
