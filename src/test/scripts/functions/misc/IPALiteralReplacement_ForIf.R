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

foo <- function(A)
{
   for( i in 1:1 ) {
     continue = TRUE;
     if( sum(A)<0 ) {
        continue = FALSE;
     }
     iter = 0;
     if( continue ) {
        iter = iter+1;
     }
     B = A+iter;
   }
   
   return(B);
}

A = matrix(1, 10, 10)
R = foo(A)

writeMM(as(R, "CsparseMatrix"), paste(args[1], "R", sep="")); 
