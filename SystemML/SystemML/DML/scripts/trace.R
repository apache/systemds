A = readMM("matrix", rows=10, cols=10, format="text");
b = trace(A);
bb = castAsScalar(b) ;
C = A * bb;
writeMM(C,"C.matrix", format="text") ;