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

#note: we use matrix here, becase Matrix created out-of-memory issues
# 'Cholmod error 'out of memory' at file ../Core/cholmod_memory.c, line 147'
# however, R still fails with 'Error: cannot allocate vector of size 3.0 Gb'

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")))

C <- A %*% B;
#C <- A %o% B; 

cmin <- min(C);

writeMM(as(cmin, "CsparseMatrix"), paste(args[2], "C", sep="")); 


