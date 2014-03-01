#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

A = seq(as.numeric(args[1]), as.numeric(args[2]));
writeMM(as(A,"CsparseMatrix"), paste(args[3], "A", sep=""), format="text")

