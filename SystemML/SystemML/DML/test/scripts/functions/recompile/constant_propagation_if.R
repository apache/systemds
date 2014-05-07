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

numrows = as.integer(args[1]);
numcols = as.integer(args[2]);

if( 1==1 )
{
   numrows = numrows + 1;
   numcols = numcols + 2;
}  

X = matrix(1, numrows, numcols);

writeMM(as(X, "CsparseMatrix"), paste(args[3], "X", sep="")); 


