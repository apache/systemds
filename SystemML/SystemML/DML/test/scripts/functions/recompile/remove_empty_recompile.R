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

X <- readMM(paste(args[1], "X.mtx", sep=""))

type = as.integer(args[2]);

R = X;

if( type==0 ){
  R = as.matrix( sum(X) );
}
if( type==1 ){
  R = round(X);
}
if( type==2 ){
  R = t(X); 
}
if( type==3 ){
  R = X*(X-1);
}
if( type==4 ){
  R = X+(X-1);
}
if( type==5 ){
  R = X%*%(X-1);
}


writeMM(as(R, "CsparseMatrix"), paste(args[3], "R", sep="")); 