#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2015
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)
library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""))
y = readMM(paste(args[1], "y.mtx", sep=""))

intercept = as.integer(args[2]);
eps = as.double(args[3]);
maxiter = as.double(args[4]);

if( intercept == 1 ){
   ones = matrix(1, nrow(X), 1); 
   X = cbind(X, ones);
}

r = -(t(X) %*% y);
p = -r;
norm_r2 = sum(r * r);
w = matrix(0, ncol(X), 1);

i = 0;
while(i < maxiter) {
	q = ((t(X) %*% (X %*% p)) + eps  * p);
	alpha = norm_r2 / ((t(p) %*% q)[1:1]);
	w = w + alpha * p;
	old_norm_r2 = norm_r2;
	r = r + alpha * q;
	norm_r2 = sum(r * r);
	beta = norm_r2 / old_norm_r2;
	p = -r + beta * p;
	i = i + 1;
}

writeMM(as(w,"CsparseMatrix"), paste(args[5], "w", sep=""))
