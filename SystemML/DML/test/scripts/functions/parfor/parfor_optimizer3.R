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

V1 <- readMM(paste(args[1], "V.mtx", sep=""))
V <- as.matrix(V1);
n <- ncol(V); 
n2 <- n/2;

R <- array(0,dim=c(1,n2))

for( i in 1:n2 )
{
   X <- V[,i];                 
   Y <- V[,n-i+1];                
   R[1,i] <- sum(X)+sum(Y);
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep="")); 
