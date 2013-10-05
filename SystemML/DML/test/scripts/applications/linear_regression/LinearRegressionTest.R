#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.applications.LinearRegressionTest.java
library("Matrix");

V = readMM("$$indir$$v.mtx");
y = readMM("$$indir$$y.mtx");
eps  = $$eps$$;
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

writeMM(w, "$$Routdir$$w");
