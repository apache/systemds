# JUnit test class: dml.test.integration.applications.GNMFTest.java
library(Matrix)

V = readMM("$$indir$$v.mtx");
W = readMM("$$indir$$w.mtx");
H = readMM("$$indir$$h.mtx");
max_iteration = $$maxiter$$;
i = 0;

Eps = 10^-8;

while(i < max_iteration) {
	H = H * ((t(W) %*% V) / (((t(W) %*% W) %*% H)+Eps)) ;
	W = W * ((V %*% t(H)) / ((W %*% (H %*% t(H)))+Eps));
	i = i + 1;
}

writeMM(as(W, "CsparseMatrix"), "$$Routdir$$w");
writeMM(as(H, "CsparseMatrix"), "$$Routdir$$h");