args = commandArgs(TRUE)

library("unbalanced")
library("Matrix")

X = paste(args[1])
y = paste(args[2])
save_as = paste(args[3])

X_under = as.matrix(ubTomek(X, y)$X)
writeMM(as(X_under, "CsparseMatrix"), save_as)
