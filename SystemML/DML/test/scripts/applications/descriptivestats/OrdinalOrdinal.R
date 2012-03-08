# JUnit test class: dml.test.integration.descriptivestats.BivariateOrdinalOrdinalTest.java
# command line invocation assuming $OO_HOME is set to the home of the R script
# Rscript $OO_HOME/OrdinalOrdinal.R $OO_HOME/in/ $OO_HOME/expected/
args <- commandArgs(TRUE)

library("Matrix")

A = readMM(paste(args[1], "A.mtx", sep=""))
B = readMM(paste(args[1], "B.mtx", sep=""))
Helper=matrix(1, 2, 1)

spearman = cor(A[,1],B[,1], method="spearman");

SpearmanHelper = spearman * Helper;
writeMM(as(t(SpearmanHelper),"CsparseMatrix"), paste(args[2], "outSpearman", sep=""), format="text");

