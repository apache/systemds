args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
colnames(X) = colnames(X, do.NULL=FALSE, prefix="C")

Y = X
for (j in 1:ncol(X)) {
  col = X[, j]
  med = quantile(col, probs=0.5, type=1, names=FALSE, na.rm=FALSE)
  q1  = quantile(col, probs=0.25, type=1, names=FALSE, na.rm=FALSE)
  q3  = quantile(col, probs=0.75, type=1, names=FALSE, na.rm=FALSE)
  iqr = q3 - q1

  # Print debug values for first column
  if (j == 1) {
    cat("R column 1:\n")
    cat("Q1: ", q1, "\n")
    cat("Median: ", med, "\n")
    cat("Q3: ", q3, "\n")
    cat("IQR: ", iqr, "\n")
  }

  if (j == 2) {
    cat("R column 2:\n")
    cat("Q1: ", q1, "\n")
    cat("Median: ", med, "\n")
    cat("Q3: ", q3, "\n")
    cat("IQR: ", iqr, "\n")
  }

  if (iqr == 0 || is.nan(iqr)) iqr = 1
  Y[, j] = (col - med) / iqr
}

writeMM(as(Y, "CsparseMatrix"), paste(args[2], "B", sep=""))
