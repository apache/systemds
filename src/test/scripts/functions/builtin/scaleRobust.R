args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")
library("caret")

X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
colnames(X) = colnames(X, do.NULL=FALSE, prefix="C")

# Robust scaling: subtract median, divide by IQR
Y = X
for (j in 1:ncol(X)) {
  med = median(X[, j])
  iqr = IQR(X[, j])
  if (iqr == 0) iqr = 1
  Y[, j] = (X[, j] - med) / iqr
}
writeMM(as(Y, "CsparseMatrix"), paste(args[2], "B", sep=""))
