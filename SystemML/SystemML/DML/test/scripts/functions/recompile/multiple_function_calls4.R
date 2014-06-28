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

V <- readMM(paste(args[1], "V.mtx", sep=""))
V1 = V-0.5;
V2 = V-0.5;

if( nrow(V)>5 ) {
   V1 = V1 + 5; 
}
if( nrow(V)>5 ) {
   V2 = V2 + 5; 
} 

R = V1+V2;
writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 