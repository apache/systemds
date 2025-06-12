library("Matrix")
library("peakRAM")

args <- commandArgs(TRUE)
options(digits=22)

result <- peakRAM({
  X = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
  colnames(X) = colnames(X, do.NULL=FALSE, prefix="C")
  Y = X

  start_time <- Sys.time()

  for (j in 1:ncol(X)) {
    col = X[, j]
    med = quantile(col, probs=0.5, type=1, names=FALSE, na.rm=FALSE)
    q1  = quantile(col, probs=0.25, type=1, names=FALSE, na.rm=FALSE)
    q3  = quantile(col, probs=0.75, type=1, names=FALSE, na.rm=FALSE)
    iqr = q3 - q1
    if (iqr == 0 || is.nan(iqr)) iqr = 1
    Y[, j] = (col - med) / iqr
  }

  end_time <- Sys.time()
  print("R overall time (s):")
  print(end_time - start_time)

  writeMM(as(Y, "CsparseMatrix"), paste(args[2], "B", sep=""))
})

print(result[, c("Elapsed_Time_sec", "Total_RAM_Used_MiB", "Peak_RAM_Used_MiB")])
