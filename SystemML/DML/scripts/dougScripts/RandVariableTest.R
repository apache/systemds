A = readMM ("dml/scripts/ex.mult.A", rows= 10, cols=5, rows_in_block = 2, columns_in_block = 2, format="text");
a = nrow(A);
b = 5;
C = Rand(rows=a, cols=b);

i = 0
max_iteration = 1

while(i < max_iteration) {
	D = t(C) %*% C ;
	i = i + 1 ;
}

writeMM(D, "scripts/RandVariableTest.D", format="text");