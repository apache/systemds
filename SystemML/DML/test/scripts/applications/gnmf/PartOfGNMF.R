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
	R = H %*% t(H);
	i = i + 1;
}

writeMM(as(R, "CsparseMatrix"), "$$Routdir$$r");
writeMM(as(H, "CsparseMatrix"), "$$Routdir$$h");
