# JUnit test class: dml.test.integration.descriptivestats.BivariateScaleCategoricalTest.java
# command line invocation assuming $SC_HOME is set to the home of the R script
# Rscript $SC_HOME/ScaleCategorical.R $SC_HOME/in/ $SC_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A = readMM(paste(args[1], "A.mtx", sep=""));
Y = readMM(paste(args[1], "Y.mtx", sep=""));

Av = A[,1];
Yv = Y[,1];

W = nrow(A);
my = mean(Yv); #sum(Yv)/W;
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

print(anova_num, digits=15);
print(anova_den, digits=15);

write(Eta, paste(args[2], "Eta", sep=""));

write(ANOVAF, paste(args[2], "AnovaF", sep=""));

write(varY, paste(args[2], "VarY", sep=""));

write(my, paste(args[2], "MeanY", sep=""));

writeMM(as(CVars,"CsparseMatrix"), paste(args[2], "CVars", sep=""), format="text");
writeMM(as(CFreqs,"CsparseMatrix"), paste(args[2], "CFreqs", sep=""), format="text");
writeMM(as(CMeans,"CsparseMatrix"), paste(args[2], "CMeans", sep=""), format="text");



