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
m <- nrow(V); 
n <- ncol(V); 

R1 <- matrix(0,m,n);
R2 <- matrix(0,m,n);

for( i in 1:n )
{
   X <- V[,i];
   R1[,i] <- X;
   R2[,i] <- X;
}   

R <- R1 + R2; 
writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep=""));
