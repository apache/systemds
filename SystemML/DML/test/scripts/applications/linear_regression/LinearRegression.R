# JUnit test class: dml.test.integration.applications.LinearRegressionTest.java
# command line invocation assuming $LR_HOME is set to the home of the R script
# Rscript $LR_HOME/LinearRegression.R $LR_HOME/in/ 0.00000001 $LR_HOME/expected/
args <- commandArgs(TRUE);

library("Matrix");

V = readMM(paste(args[1], "v.mtx", sep=""));
y = readMM(paste(args[1], "y.mtx", sep=""));

eps  = as.double(args[2]);

r = -(t(V) %*% y);
p = -r;
norm_r2 = sum(r * r);
w = 0;

max_iteration = 3;
i = 0;
while(i < max_iteration) {
	q = ((t(V) %*% (V %*% p)) + eps * p);
	alpha = norm_r2 / ((t(p) %*% q)[1:1]);
	w = w + alpha * p;
	old_norm_r2 = norm_r2;
	r = r + alpha * q;
	norm_r2 = sum(r * r);
	beta = norm_r2 / old_norm_r2;
	p = -r + beta * p;
	i = i + 1;
}

writeMM(w, paste(args[3], "w", sep=""));

