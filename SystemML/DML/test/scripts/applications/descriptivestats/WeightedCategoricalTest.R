# JUnit test class: dml.test.integration.descriptivestats.UnivariateStatsTest.java
# command line invocation assuming $C_HOME is set to the home of the R script
# Rscript $C_HOME/Categorical.R $C_HOME/in/ $C_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

#library("batch")
library("Matrix")
# Usage: R --vanilla -args Xfile X < DescriptiveStatistics.R

#parseCommandArgs()
######################

V = readMM(paste(args[1], "vector.mtx", sep=""))
W = readMM(paste(args[1], "weight.mtx", sep=""))

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

writeMM(as(t(Nc),"CsparseMatrix"), paste(args[2], "Nc", sep=""), format="text");
write(R, paste(args[2], "R", sep=""));
writeMM(as(t(Pc),"CsparseMatrix"), paste(args[2], "Pc", sep=""), format="text");
writeMM(as(t(C),"CsparseMatrix"), paste(args[2], "C", sep=""), format="text");
writeMM(as(t(Mode),"CsparseMatrix"), paste(args[2], "Mode", sep=""), format="text");
