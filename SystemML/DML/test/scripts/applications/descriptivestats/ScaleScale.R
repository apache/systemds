#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.CategoricalCategoricalTest.java
# command line invocation assuming $SS_HOME is set to the home of the R script
# Rscript $SS_HOME/ScaleScale.R $SS_HOME/in/ $SS_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""))
Y = readMM(paste(args[1], "Y.mtx", sep=""))

# cor.test returns a list containing t-statistic, df, p-value, and R
cort = cor.test(X[,1], Y[,1]);

R = as.numeric(cort[4]);

write(R, paste(args[2], "PearsonR", sep=""));
