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

V <- readMM(paste(args[1], "V.mtx", sep=""))
n <- ncol(V); 

R <- array(0,dim=c(n,1))

for( i in 1:n )
{
   X <- V[2:nrow(V),i];                 
   R[i,1] <- sum(X %*% t(X));
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep="")); 