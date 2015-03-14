#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2014
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

X <- as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
W <- as.matrix(readMM(paste(args[1], "W.mtx", sep="")))

Nt = nrow(X);
num_classes = ncol(W)
n = ncol(X);

b = W[n+1,]
ones = matrix(1, Nt, 1)
scores = X %*% W[1:n,] + ones %*% b;

predicted_y = max.col(scores,ties.method="last")

writeMM(as(predicted_y, "CsparseMatrix"), paste(args[2], "predicted_y", sep="")); 


