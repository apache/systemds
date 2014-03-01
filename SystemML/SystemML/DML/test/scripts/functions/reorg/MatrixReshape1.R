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

X=readMM(paste(args[1], "X.mtx", sep=""))
Y=matrix(t(X),nrow=as.numeric(args[2]),ncol=as.numeric(args[3]),byrow=TRUE)
writeMM(as(Y,"CsparseMatrix"), paste(args[4], "Y", sep=""), format="text")




