#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < DescriptiveStatistics.R

#parseCommandArgs()
######################

V = readMM(file="$$indir$$vector.mtx")
W = readMM(file="$$indir$$weight.mtx")
Helper=matrix(1, 2, 1)

tab = table(rep(V[,1],W[,1]))
cat = t(as.numeric(names(tab)))
Nc = t(as.vector(tab))

# the number of categories of a categorical variable
R = length(Nc)

# total count
s = sum(Nc)

# percentage values of each categorical compare to the total case number
Pc = Nc / s

# all categorical values of a categorical variable
#C = t(as.matrix(as.numeric(Nc > 0)))
C= (Nc > 0)

# mode
mx = max(Nc)
Mode = (Nc == mx)

RHelper=R*Helper
writeMM(as(t(Nc),"CsparseMatrix"), "$$Routdir$$Nc_weight", format="text");
writeMM(as(t(RHelper),"CsparseMatrix"), "$$Routdir$$R_weight", format="text");
writeMM(as(t(Pc),"CsparseMatrix"), "$$Routdir$$Pc_weight", format="text");
writeMM(as(t(C),"CsparseMatrix"), "$$Routdir$$C_weight", format="text");
writeMM(as(t(Mode),"CsparseMatrix"), "$$Routdir$$Mode_weight", format="text");
