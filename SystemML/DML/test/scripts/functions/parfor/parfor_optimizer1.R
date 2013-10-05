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

#NOTES MB: readMM returns an obj inherited from matrix
# (it seams like it internally uses lists, which makes
# is very slow if there are multiple passes over the data). 
# adding 'V <- as.matrix(V1)' by more than a factor of 10.
# However, this will always result in a dense matrix. 

V1 <- readMM(paste(args[1], "V.mtx", sep=""))
V <- as.matrix(V1);

m <- nrow(V);
n <- ncol(V); 
W <- m;

R <- array(0,dim=c(n,n))

for( i in 1:(n-1) )
{
   X <- V[ ,i];                 
      
   for( j in (i+1):n )  
   {
      Y <- V[ ,j];  
      R[i,j] <- cor(X, Y)  
#      print(R[i,j]);
   }
}   

writeMM(as(R, "CsparseMatrix"), paste(args[2], "Rout", sep=""));
