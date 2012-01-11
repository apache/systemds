#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < DescriptiveStatistics.R

#parseCommandArgs()
######################

V = readMM(file="$$indir$$vector.mtx")
Helper=matrix(1, 2, 1)

tab = table(V[,1])
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
writeMM(as(t(Nc),"CsparseMatrix"), "$$Routdir$$Nc", format="text");
writeMM(as(t(RHelper),"CsparseMatrix"), "$$Routdir$$R", format="text");
writeMM(as(t(Pc),"CsparseMatrix"), "$$Routdir$$Pc", format="text");
writeMM(as(t(C),"CsparseMatrix"), "$$Routdir$$C", format="text");
writeMM(as(t(Mode),"CsparseMatrix"), "$$Routdir$$Mode", format="text");
