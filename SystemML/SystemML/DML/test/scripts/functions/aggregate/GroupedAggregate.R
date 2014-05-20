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
fn = as.integer(args[2]);

if( fn==0 )
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=sum)[,2]
}

if( fn==1 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=length)[,2]
}

if( fn==2 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=mean)[,2]
}

if( fn==3 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=var)[,2]
}

if( fn==4 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=moment, order=3, central=TRUE)[,2]
}

if( fn==5 ) 
{
   C = aggregate(as.vector(A), by=list(as.vector(B)), FUN=moment, order=4, central=TRUE)[,2]
}

writeMM(as(C, "CsparseMatrix"), paste(args[3], "C", sep="")); 