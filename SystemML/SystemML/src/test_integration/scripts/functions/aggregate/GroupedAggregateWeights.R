#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2014
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)

library("Matrix")
library("moments")

A <- as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
B <- as.matrix(readMM(paste(args[1], "B.mtx", sep="")));
C <- as.matrix(readMM(paste(args[1], "C.mtx", sep="")));
fn = as.integer(args[2]);

if( nrow(A)==1 & ncol(A)>1 ){ #row vector
   A = t(A);
}

if( fn==0 )
{
   #special case weights
   D = aggregate(as.vector(A*C), by=list(as.vector(B)), FUN=sum)[,2]
}

if( fn==1 ) 
{
   #special case weights
   D = aggregate(as.vector(C), by=list(as.vector(B)), FUN=sum)[,2]
}

if( fn==2 ) 
{
   #special case weights
   D1 = aggregate(as.vector(A*C), by=list(as.vector(B)), FUN=sum)[,2]
	 D2 = aggregate(as.vector(C), by=list(as.vector(B)), FUN=sum)[,2]
   D = D1/D2;
}

if( fn==3 ) 
{
   D = aggregate(as.vector(A*C), by=list(as.vector(B)), FUN=var)[,2]
}

if( fn==4 ) 
{
   D = aggregate(as.vector(A*C), by=list(as.vector(B)), FUN=moment, order=3, central=TRUE)[,2]
}

if( fn==5 ) 
{
   D = aggregate(as.vector(A*C), by=list(as.vector(B)), FUN=moment, order=4, central=TRUE)[,2]
}

writeMM(as(D, "CsparseMatrix"), paste(args[3], "D", sep="")); 