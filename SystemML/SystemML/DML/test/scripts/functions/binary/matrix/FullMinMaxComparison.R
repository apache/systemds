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

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")))

if( as.integer(args[2])==1 ){
  # MIN
  if( nrow(A)>1 | nrow(B)>1 ){
     if( nrow(B)>nrow(A) ) {
       C <- pmin(B, A);
     }  
     else {
       C <- pmin(A, B);
     }
  }else{
     C <- min(A, B);
  }    
} else{
  # MAX  
  if( nrow(A)>1 | nrow(B)>1 ){
     if( nrow(B)>nrow(A) ){
       C <- pmax(B, A);
     }else{
       C <- pmax(A, B);
     }  
  }else{
     C <- max(A, B);
  } 
}

writeMM(as(as.matrix(C), "CsparseMatrix"), paste(args[3], "C", sep="")); 


