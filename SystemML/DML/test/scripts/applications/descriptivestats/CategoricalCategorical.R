# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $CC_HOME is set to the home of the R script
# Rscript $CC_HOME/CategoricalCategorical.R $CC_HOME/in/ $CC_HOME/expected/
args <- commandArgs(TRUE)

library("Matrix")

A = readMM(paste(args[1], "A.mtx", sep=""));
B = readMM(paste(args[1], "B.mtx", sep=""));

Helper=matrix(1, 2, 1);

F = table(A[,1],B[,1]);

# chisq.test returns a list containing statistic, p-value, etc.
cst = chisq.test(F);

# get the chi-squared coefficient from the list
chi_squared = as.numeric(cst[1]);
pValue = as.numeric(cst[3]);

PValueHelper = pValue * Helper;
writeMM(as(t(PValueHelper),"CsparseMatrix"), paste(args[2], "PValue", sep=""), format="text");

q = min(dim(F));
W = sum(F);
cramers_v = sqrt(chi_squared/(W*(q-1)));

CramersVHelper = cramers_v * Helper;
writeMM(as(t(CramersVHelper),"CsparseMatrix"), paste(args[2], "CramersV", sep=""), format="text");

