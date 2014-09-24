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

V = as.matrix(readMM(paste(args[1], "V.mtx", sep="")))
n = ncol(V); 
R = matrix(0, 1, n);

iter = 1;
while( iter <= 3 )
{
   if( as.integer(args[3])==1 )
   {
      vx = matrix(1,nrow(V),1)*iter;
      V = cbind(V, vx);
      rx = matrix(0,1,1);
      R = cbind(R, rx);
   }
   
   for( i in 1:ncol(V) )
   {
      Xi = V[,i];
      R[1,i] = R[1,i] + sum(Xi);
   }
   
   iter = iter+1;
}

writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 