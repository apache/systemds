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

factorial <- function(arr, pos){
	if(pos == 1){
     arr[1, pos] = 1
	} else {
		arr = factorial(arr, pos-1)
		arr[1, pos] = pos * arr[1, pos-1]
	}

  return(arr);	
}

n = as.integer(args[1])
arr = matrix(0, 1, n)
arr = factorial(arr, n)

R = matrix(0, 1, n);
for(i in 1:n)
{
   R[1,i] = arr[1, i];
}

writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 


