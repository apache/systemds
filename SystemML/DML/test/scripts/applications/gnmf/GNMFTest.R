# JUnit test class: dml.test.integration.applications.GNMFTest.java
library(Matrix)

V = readMM(".\\test\\scripts\\applications\\gnmf\\in\\v.mtx");
W = readMM(".\\test\\scripts\\applications\\gnmf\\in\\w.mtx");
H = readMM(".\\test\\scripts\\applications\\gnmf\\in\\h.mtx");
max_iteration = 3;
i = 0;

Eps = 10^-8;

while(i < max_iteration) {
	H = H * ((t(W) %*% V) / (((t(W) %*% W) %*% H)+Eps)) ;
	W = W * ((V %*% t(H)) / ((W %*% (H %*% t(H)))+Eps));
	i = i + 1;
}

writeMM(as(W, "CsparseMatrix"), ".\\test\\scripts\\applications\\gnmf\\expected\\w");
writeMM(as(H, "CsparseMatrix"), ".\\test\\scripts\\applications\\gnmf\\expected\\h");