#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2015
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
A <- floor(A);

IA = (A != 0) * seq (1, nrow (A), 1);
IA = matrix (IA, (nrow (A) * ncol(A)), 1, byrow = FALSE);
VA = matrix ( A, (nrow (A) * ncol(A)), 1, byrow = FALSE);
#IA = removeEmpty (target = IA, margin = "rows");
#VA = removeEmpty (target = VA, margin = "rows");
Btmp1 = table (IA, VA);
Btmp2 = as.matrix(as.data.frame.matrix(Btmp1));

#remove first row and column (0 values, see missing removeEmpty)
B = Btmp1[2:nrow(Btmp2),2:ncol(Btmp2)];

writeMM(as(B, "CsparseMatrix"), paste(args[2], "B", sep="")); 


