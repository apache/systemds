# JUnit test class: dml.test.integration.applications.L2SVMTest.java
# command line invocation assuming $L2SVM_HOME is set to the home of the R script
# Rscript $L2SVM_HOME/L2SVM.R $L2SVM_HOME/in/ 0.00000001 1 $L2SVM_HOME/expected/

args <- commandArgs(TRUE)
library("Matrix")

X = readMM(paste(args[1], "X.mtx", sep=""));
Y = readMM(paste(args[1], "Y.mtx", sep=""));

epsilon = as.double(args[2]);
lambda = 1;

N = nrow(X)
D = ncol(X)

w = matrix(0,D,1)

g_old = t(X) %*% Y
s = g_old

continue = TRUE
while(continue){
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
}

writeMM(as(w,"CsparseMatrix"), paste(args[4], "w", sep=""));
