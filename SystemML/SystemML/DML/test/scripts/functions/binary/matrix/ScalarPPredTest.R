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

A1 <- readMM(paste(args[1], "A.mtx", sep=""))
A <- as.matrix(A1);

type = as.integer(args[2])
constant = as.double(args[3]);

if( type == 0 )
{
   B = (A > constant)
}
if( type == 1 )
{
   B = (A < constant)
}
if( type == 2 )
{
   B = (A == constant)
}
if( type == 3 )
{
   B = (A != constant)
}
if( type == 4 )
{
   B = (A >= constant)
}
if( type == 5 )
{
   B = (A <= constant)
}


writeMM(as(B, "CsparseMatrix"), paste(args[4], "B", sep="")); 