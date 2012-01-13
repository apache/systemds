#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < ScaleCategoricalTest.R

#parseCommandArgs()
######################

A = readMM(file="$$indir$$A.mtx");
Y = readMM(file="$$indir$$Y.mtx");
Helper=matrix(1, 2, 1)

Av = A[,1];
Yv = Y[,1];

W = nrow(A);
my = mean(Yv); #sum(Yv)/W;
varY = var(Yv);

CFreq = as.matrix(table(Av)); 
CMeans = as.matrix(aggregate(Yv, by=list(Av), "mean")$x);
CVars = as.matrix(aggregate(Yv, by=list(Av), "var")$x);

# number of categories
R = nrow(CFreq);

Eta = sqrt(1 - ( sum((CFreq-1)*CVars) / ((W-1)*varY) ));

anova_num = sum( (CFreq*(CMeans-my)^2) )/(R-1);
anova_den = sum( (CFreq-1)*CVars )/(W-R);
ANOVAF = anova_num/anova_den;

#print(anova_num, digits=15);
#print(anova_den, digits=15);

#######################

EtaHelper = Eta * Helper;
writeMM(as(t(EtaHelper),"CsparseMatrix"), "$$Routdir$$outEta", format="text");
#print(EtaHelper, digits=15);

AnovaFHelper = ANOVAF * Helper;
writeMM(as(t(AnovaFHelper),"CsparseMatrix"), "$$Routdir$$outAnovaF", format="text");
#print(AnovaFHelper, digits=15);

VarYHelper = varY * Helper;
writeMM(as(t(VarYHelper),"CsparseMatrix"), "$$Routdir$$outVarY", format="text");
#print(VarYHelper, digits=15);

MeanYHelper = my * Helper;
writeMM(as(t(MeanYHelper),"CsparseMatrix"), "$$Routdir$$outMeanY", format="text");
#print(MeanYHelper, digits=15);

writeMM(as(CVars,"CsparseMatrix"), "$$Routdir$$outCatVars", format="text");
writeMM(as(CFreq,"CsparseMatrix"), "$$Routdir$$outCatFreqs", format="text");
writeMM(as(CMeans,"CsparseMatrix"), "$$Routdir$$outCatMeans", format="text");

#print(CVars, digits=15);
#print(CFreq, digits=15);
#print(CMeans, digits=15);


