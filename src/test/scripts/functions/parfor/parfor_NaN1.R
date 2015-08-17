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


nr = as.integer(args[1]);
xr = as.integer(args[2]);
NaNval = 0/0;

R = matrix(0, nr, nr); 
R[1:xr,] = matrix(NaNval, xr, nr);

for( i in 1:nr )
{
   R[i,i] = i^2 + 7;           
}   

writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep=""));
