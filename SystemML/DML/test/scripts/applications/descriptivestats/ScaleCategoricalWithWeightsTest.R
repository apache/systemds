#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.BivariateScaleCategoricalTest.java
# command line invocation assuming $SC_HOME is set to the home of the R script
# Rscript $SC_HOME/ScaleCategorical.R $SC_HOME/in/ $SC_HOME/expected/

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")
# Usage: R --vanilla -args Xfile X < ScaleCategoricalTest.R

#parseCommandArgs()
######################
Atemp = readMM(paste(args[1], "A.mtx", sep=""));
Ytemp = readMM(paste(args[1], "Y.mtx", sep=""));
WM = readMM(paste(args[1], "WM.mtx", sep=""));

Yv=rep(Ytemp[,1],WM[,1])
Av=rep(Atemp[,1],WM[,1])

W = sum(WM);
my = sum(Yv)/W;
varY = var(Yv);

CFreqs = as.matrix(table(Av)); 
CMeans = as.matrix(aggregate(Yv, by=list(Av), "mean")$x);
CVars = as.matrix(aggregate(Yv, by=list(Av), "var")$x);

# number of categories
R = nrow(CFreqs);

Eta = sqrt(1 - ( sum((CFreqs-1)*CVars) / ((W-1)*varY) ));

anova_num = sum( (CFreqs*(CMeans-my)^2) )/(R-1);
anova_den = sum( (CFreqs-1)*CVars )/(W-R);
ANOVAF = anova_num/anova_den;

print(W, digits=15);
print(R, digits=15);
print(anova_num, digits=15);
print(anova_den, digits=15);

#######################

write(Eta, paste(args[2], "Eta", sep=""));

write(ANOVAF, paste(args[2], "AnovaF", sep=""));

write(varY, paste(args[2], "VarY", sep=""));

write(my, paste(args[2], "MeanY", sep=""));

writeMM(as(CVars,"CsparseMatrix"), paste(args[2], "CVars", sep=""), format="text");
writeMM(as(CFreqs,"CsparseMatrix"), paste(args[2], "CFreqs", sep=""), format="text");
writeMM(as(CMeans,"CsparseMatrix"), paste(args[2], "CMeans", sep=""), format="text");



