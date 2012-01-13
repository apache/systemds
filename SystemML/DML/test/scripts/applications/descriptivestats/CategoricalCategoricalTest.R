#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < NominalNominalTest.R

#parseCommandArgs()
######################

A = readMM(file="$$indir$$A.mtx")
B = readMM(file="$$indir$$B.mtx")
Helper=matrix(1, 2, 1)

F = table(A[,1],B[,1]);

# chisq.test returns a list containing statistic, p-value, etc.
cst = chisq.test(F);

# get the chi-squared coefficient from the list
# chi_squared = as.numeric(cst[1]);
chi_squared = as.numeric(cst[1]);
pValue = as.numeric(cst[3]);

PValueHelper = pValue * Helper;
writeMM(as(t(PValueHelper),"CsparseMatrix"), "$$Routdir$$outPValue", format="text");

#######################

q = min(dim(F));
W = sum(F);
cramers_v = sqrt(chi_squared/(W*(q-1)));

CramersVHelper = cramers_v * Helper;
writeMM(as(t(CramersVHelper),"CsparseMatrix"), "$$Routdir$$outCramersV", format="text");

