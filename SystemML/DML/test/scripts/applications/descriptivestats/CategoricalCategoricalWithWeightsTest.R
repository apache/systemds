#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < NominalNominalWithWeightsTest.R

#parseCommandArgs()
######################

A = readMM(file="$$indir$$A.mtx");
B = readMM(file="$$indir$$B.mtx");
WM = readMM(file="$$indir$$WM.mtx");
Helper=matrix(1, 2, 1)

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

PValueHelper = pValue * Helper;
writeMM(as(t(PValueHelper),"CsparseMatrix"), "$$Routdir$$outPValue", format="text");

#######################

q = min(dim(F));
W = sum(F);
cramers_v = sqrt(chi_squared/(W*(q-1)));

CramersVHelper = cramers_v * Helper;
writeMM(as(t(CramersVHelper),"CsparseMatrix"), "$$Routdir$$outCramersV", format="text");

