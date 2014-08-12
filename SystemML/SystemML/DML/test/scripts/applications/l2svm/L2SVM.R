#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.applications.L2SVMTest.java
# command line invocation assuming $L2SVM_HOME is set to the home of the R script
# Rscript $L2SVM_HOME/L2SVM.R $L2SVM_HOME/in/ 0.00000001 1 100 $L2SVM_HOME/expected/

args <- commandArgs(TRUE)
library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""));
Y = readMM(paste(args[1], "Y.mtx", sep=""));

check_min = min(Y)
check_max = max(Y)
num_min = sum(Y == check_min)
num_max = sum(Y == check_max)
if(num_min + num_max != nrow(Y)){ 
	print("please check Y, it should contain only 2 labels") 
}else{
	if(check_min != -1 | check_max != +1) 
		Y = 2/(check_max - check_min)*Y - (check_min + check_max)/(check_max - check_min)
}

intercept = as.integer(args[2]);
epsilon = as.double(args[3]);
lambda = as.double(args[4]);
maxiterations = as.integer(args[5]);

N = nrow(X)
D = ncol(X)

if (intercept == 1) {
	ones  = matrix(1,N,1)
	X = cbind(X, ones);
}

num_rows_in_w = D
if(intercept == 1){
	num_rows_in_w = num_rows_in_w + 1
}
w = matrix(0, num_rows_in_w, 1)

g_old = t(X) %*% Y
s = g_old

Xw = matrix(0,nrow(X),1)
iter = 0
continue = TRUE
while(continue && iter < maxiterations){
	t = 0
	Xd = X %*% s
	wd = lambda * sum(w * s)
	dd = lambda * sum(s * s)
	continue1 = TRUE
	while(continue1){
		tmp_Xw = Xw + t*Xd
		out = 1 - Y * (tmp_Xw)
		sv = which(out > 0)
		g = wd + t*dd - sum(out[sv] * Y[sv] * Xd[sv])
		h = dd + sum(Xd[sv] * Xd[sv])
		t = t - g/h
		continue1 = (g*g/h >= 1e-10)
	}
	
	w = w + t*s
	Xw = Xw + t*Xd
		
	out = 1 - Y * (X %*% w)
	sv = which(out > 0)
	obj = 0.5 * sum(out[sv] * out[sv]) + lambda/2 * sum(w * w)
	g_new = t(X[sv,]) %*% (out[sv] * Y[sv]) - lambda * w
	
	print(paste("OBJ : ", obj))

	continue = (t*sum(s * g_old) >= epsilon*obj)
	
	be = sum(g_new * g_new)/sum(g_old * g_old)
	s = be * s + g_new
	g_old = g_new
	
	iter = iter + 1
}

writeMM(as(w,"CsparseMatrix"), paste(args[6], "w", sep=""));
