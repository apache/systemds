# JUnit test class: dml.test.integration.descriptivestats.BivariateOrdinalOrdinalTest.java
# command line invocation assuming $OO_HOME is set to the home of the R script
# Rscript $OO_HOME/OrdinalOrdinal.R $OO_HOME/in/ $OO_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A = readMM(paste(args[1], "A.mtx", sep=""))
B = readMM(paste(args[1], "B.mtx", sep=""))

spearman = cor(A[,1],B[,1], method="spearman");

#paste("R value", spearman);

write(spearman, paste(args[2], "Spearman", sep=""));

