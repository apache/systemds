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

epsilon = as.double(args[2]);
lambda = as.double(args[3]);
maxiterations = as.integer(args[4]);

N = nrow(X)
D = ncol(X)

#checking Y's correctness
max_y = max(Y)
min_y = min(Y)
sum_abs_y = sum(abs(Y))

if(max_y == 1 && min_y == -1 && sum_abs_y == N){
	w = matrix(0,D,1)

	g_old = t(X) %*% Y
	s = g_old

	iter = 0
	continue = TRUE
	while(continue && iter < maxiterations){
		t = 0
		Xd = X %*% s
		wd = lambda * sum(w * s)
		dd = lambda * sum(s * s)
		continue1 = TRUE
		while(continue1){
			tmp_w = w + t*s
			out = 1 - Y * (X %*% tmp_w)
			sv = which(out > 0)
			g = wd + t*dd - sum(out[sv] * Y[sv] * Xd[sv])
			h = dd + sum(Xd[sv] * Xd[sv])
			t = t - g/h
			continue1 = (g*g/h >= 1e-10)
		}
	
		w = w + t*s
	
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

	writeMM(as(w,"CsparseMatrix"), paste(args[5], "w", sep=""));
}else{
	print("Training labels (Y) can only be -1 or +1 (binary classification). "
		  + "Please ensure that before invoking L2SVM.dml");
}
