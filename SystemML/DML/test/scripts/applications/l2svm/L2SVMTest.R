# JUnit test class: dml.test.integration.applications.L2SVMTest.java
library("Matrix")

X = readMM("$$indir$$X.mtx")
Y = readMM("$$indir$$Y.mtx")
epsilon = $$eps$$
lambda = $$lambda$$

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

writeMM(as(w,"CsparseMatrix"), "$$Routdir$$w", format="text");
