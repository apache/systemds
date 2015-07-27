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

#without weights (assumes weights of 1)
m = nrow(A);
S = sort(A)
q25d=m*0.25
q75d=m*0.75
q25i=ceiling(q25d)
q75i=ceiling(q75d)
iqm = sum(S[(q25i+1):q75i])
iqm = iqm + (q25i-q25d)*S[q25i] - (q75i-q75d)*S[q75i]
iqm = iqm/(m*0.5)

miqm = as.matrix(iqm);

writeMM(as(miqm, "CsparseMatrix"), paste(args[3], "R", sep="")); 


