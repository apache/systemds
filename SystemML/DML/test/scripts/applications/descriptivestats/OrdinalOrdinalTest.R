#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < OrdinalOrdinalTest.R

#parseCommandArgs()
######################

A = readMM(file="$$indir$$A.mtx")
B = readMM(file="$$indir$$B.mtx")
Helper=matrix(1, 2, 1)

spearman = cor(A[,1],B[,1], method="spearman");

SpearmanHelper = spearman * Helper;
writeMM(as(t(SpearmanHelper),"CsparseMatrix"), "$$Routdir$$outSpearman", format="text");

