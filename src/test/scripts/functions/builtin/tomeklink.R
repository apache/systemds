args = commandArgs(TRUE)

library("unbalanced")
library("Matrix")


# TODO test if works with such inputs

# X = paste(args[1])
# y = paste(args[2])
# save_as = paste(args[3])

# TODO remove
X = matrix(0, 5, 5)
X[sample(length(X), size = 14)] <- rep(1:9, length=14)
y = c(0,0,0,1,1)

X_under = as.matrix(ubTomek(X, y)$X)
writeMM(as(X_under, "CsparseMatrix"), save_as)
