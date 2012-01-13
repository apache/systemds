# usage: X = Rand(...)
# available parameters: rows, cols, rows_in_block, columns_in_block, min, max, sparsity, pdf

# generate a random scalar
A = Rand(min=0.0, max=10.0);
# generate a random vector
B = Rand(rows=100);
# generate a random matrix
C = Rand(rows=100, cols=100, sparsity=0.5);

D = C %*% B;
E = D * A;
writeMM(E, "scripts/Rand.E", format="text");