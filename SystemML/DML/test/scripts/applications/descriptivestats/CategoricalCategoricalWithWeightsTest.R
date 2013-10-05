#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $CC_HOME is set to the home of the R script
# Rscript $CC_HOME/CategoricalCategoricalWithWeightsTest.R $CC_HOME/in/ $CC_HOME/expected/
# Usage: R --vanilla -args Xfile X < CategoricalCategoricalWithWeightsTest.R

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

#parseCommandArgs()
######################

print(commandArgs(TRUE)[1])

A = readMM(paste(args[1], "A.mtx", sep=""));
B = readMM(paste(args[1], "B.mtx", sep=""));
WM = readMM(paste(args[1], "WM.mtx", sep=""));

Av = A[,1];
Bv = B[,1];
WMv = WM[,1];

# create a data frame with vectors A, B, WM
df = data.frame(Av,Bv,WMv);

# contingency table with weights
F = xtabs ( WMv ~ Av + Bv, df);

# chisq.test returns a list containing statistic, p-value, etc.
cst = chisq.test(F);

# get the chi-squared coefficient from the list
chi_squared = as.numeric(cst[1]);
pValue = as.numeric(cst[3]);

write(pValue, paste(args[2], "PValue", sep=""));

#######################

q = min(dim(F));
W = sum(F);
cramers_v = sqrt(chi_squared/(W*(q-1)));

write(cramers_v, paste(args[2], "CramersV", sep=""));

