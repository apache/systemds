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

A <- readMM(paste(args[1], "A.mtx", sep=""))
B <- readMM(paste(args[1], "B.mtx", sep=""))

type = as.integer(args[2])

if( type == 0 )
{
   C = (A > B)
}
if( type == 1 )
{
   C = (A < B)
}
if( type == 2 )
{
   C = (A == B)
}
if( type == 3 )
{
   C = (A != B)
}
if( type == 4 )
{
   C = (A >= B)
}
if( type == 5 )
{
   C = (A <= B)
}


writeMM(as(C, "CsparseMatrix"), paste(args[3], "C", sep="")); 