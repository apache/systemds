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

library(Matrix);

A = read.csv(args[1], header=as.logical(args[2]), sep=args[3]);
A[is.na(A)] = 0;
x =  sum(A);
write(x, args[4]);

