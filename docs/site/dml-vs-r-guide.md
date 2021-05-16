---
layout: site
title: R to DML 
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

To ease the prototyping of the `dml` scripts from its `R` counterparts, this
guide covers various practical functions or operations.

NOTE: This document is still a work in progress.

## Table of Contents

* [Multiple outputs](#multiple-outputs)
* [Order function](#order-function)
* [Read function](#read-function)
* [`Write` function](#write-function)

### Multiple-outputs

```dml
# dml

A = rand(rows=10, cols=10)
C = t(A) %*% A
[evalues, evectors] = eigen(C);
```

```R
# R

A = rand(rows=10, cols=10)
C = t(A) %*% A
R <- eigen(C);
evalues = R$values;
evectors = R$vectors;
```

### `order`-function

```dml
# dml

decreasing_Idx = order(target=evalues,by=1,decreasing=TRUE,index.return=TRUE);
diagmat = table(seq(1,D),decreasing_Idx);
```

```R
# R

decreasing_Idx = order(as.vector(evalues), decreasing=TRUE);
diagmat = table(seq(1,D), decreasing_Idx);
```

### `Read`-function

```dml
# dml

A = read("")
# A = read($INPUT)
```

```R
# R

# args[1] will the relative directory path
A = readMM(paste(args[1], "A.mtx", sep=""))
```

### `Write`-function

```dml
# dml
ofmt = "TEXT"

write(evalues, "evalues", format=ofmt)
```

```R
# R

# Here args[2] will be a string denoting output directory
writeMM(as(evalues, "CsparseMatrix"), paste(args[2],"evalues", sep=""));
```

### Recipes

#### Construct a matrix (sparse)

(rowIndex, colIndex, values) triplets

```dml
I = matrix ("1 3 3", rows = 3, cols = 1)
J = matrix ("2 3 4", rows = 3, cols = 1)
V = matrix ("10 20 30", rows = 3, cols = 1)

M = table (I, J, V)
```

#### Find and remove duplicates in columns or rows

##### Assuming values are sorted

```dml
X = matrix ("1 2 2 3 5 6 7 8 9 9", rows = 10, cols = 1)

# Compare the current value with the next value
I = rbind (matrix (1,1,1), (X[1:nrow (X)-1,] != X[2:nrow (X),]))
# Select only the unique items
res = removeEmpty (target = X, margin = "rows", select = I)
```

##### Values may not be sorted in order

Method 1:

```dml
X = matrix ("1 8 2 3 9 7 6 5 2 9", rows = 10, cols = 1)

# group and count duplicates
I = aggregate (target = X, groups = X[,1], fn = "count")
# select groups
res = removeEmpty (target = seq (1, max (X[,1])), margin = "rows", select = (I != 0))
```

Method 2:

First order and then remove duplicates

```dml
X = matrix ("3 2 1 3 3 4 5 10", rows = 8, cols = 1)

X = order (target = X, by = 1)
I = rbind (matrix (1,1,1), (X[1:nrow (X)-1,] != X[2:nrow (X),]))
res = removeEmpty (target = X, margin = "rows", select = I)
```

#### Set based indexing

Given a matrix X, with a indicator matrix I with indices into X. Use I to perform an operation
on X. For eg., add a value 10 to the cells (in X) indicated by I.

```dml
X = matrix (1, rows = 1, cols = 100)
J = matrix ("10 20 25 26 28 31 50 67 79", rows = 1, cols = 9)

res = X + table (matrix (1, rows = 1, cols = ncol (J)), J, 10)
```

#### Group by aggregate using Linear Algebra

Given a matrix PCV as (Position, Category, Value), sort PCV by category, and within each category
by value in descending order.

- create indicator vector for category changes
- create distinct categories, and
- perform linear algebra operations.

```dml
# category data
C = matrix ('50 40 20 10 30 20 40 20 30', rows = 9, cols = 1)
# value data
V = matrix ('20 11 49 33 94 29 48 74 57', rows = 9, cols = 1)

# 1. PCV representation
PCV = cbind (cbind (seq (1, nrow (C), 1), C), V)
PCV = order (target = PCV, by = 3, decreasing = TRUE,  index.return = FALSE)
PCV = order (target = PCV, by = 2, decreasing = FALSE, index.return = FALSE)

# 2. Find all rows of PCV where the category has a new value, in comparison to
# the previous row

is_new_C = matrix (1, rows = 1, cols = 1);
if (nrow (C) > 1) {
  is_new_C = rbind (is_new_C, (PCV [1:nrow(C) - 1, 2] < PCV [2:nrow(C), 2]));
}

# 3. Associate each category with its index

index_C = cumsum (is_new_C);                                                          # cumsum

# 4. For each category, compute:
#   - the list of distinct categories
#   - the maximum value for each category
#   - 0-1 aggregation matrix that adds records of the same category

distinct_C  = removeEmpty (target = PCV [, 2], margin = "rows", select = is_new_C);
max_V_per_C = removeEmpty (target = PCV [, 3], margin = "rows", select = is_new_C);
C_indicator = table (index_C, PCV [, 1], max (index_C), nrow (C));                    # table

# 5. Perform aggregation, here sum values per category
sum_V_per_C = C_indicator %*% V
```

### Invert lower triangular matrix

In this example, we invert a lower triangular matrix using the following divide-and-conquer approach.
Given lower triangular matrix L, we compute its inverse X which is also lower triangular by splitting
both matrices in the middle into 4 blocks (in a 2x2 fashion), and multiplying them together to get
the identity matrix:

$$
\begin{equation}
L \text{ %*% } X = \left(\begin{matrix} L_1 & 0 \\ L_2 & L_3 \end{matrix}\right)
\text{ %*% } \left(\begin{matrix} X_1 & 0 \\ X_2 & X_3 \end{matrix}\right)
= \left(\begin{matrix} L_1 X_1 & 0 \\ L_2 X_1 + L_3 X_2 & L_3 X_3 \end{matrix}\right)
= \left(\begin{matrix} I & 0 \\ 0 & I \end{matrix}\right)
\nonumber
\end{equation}
$$

If we multiply blockwise, we get three equations: 

$$
\begin{equation}
L1 \text{ %*% } X1 = 1\\ 
L3 \text{ %*% } X3 = 1\\
L2 \text{ %*% } X1 + L3 \text{ %*% } X2 = 0\\
\end{equation}
$$

Solving these equation gives the following formulas for X:

$$
\begin{equation}
X1 = inv(L1) \\
X3 = inv(L3) \\
X2 = - X3 \text{ %*% } L2 \text{ %*% } X1 \\
\end{equation}
$$

If we already recursively inverted L1 and L3, we can invert L2.  This suggests an algorithm
that starts at the diagonal and iterates away from the diagonal, involving bigger and bigger
blocks (of size 1, 2, 4, 8, etc.)  There is a logarithmic number of steps, and inside each
step, the inversions can be performed in parallel using a parfor-loop.

Function "invert_lower_triangular" occurs within more general inverse operations and matrix decompositions.
The divide-and-conquer idea allows to derive more efficient algorithms for other matrix decompositions.

```dml
invert_lower_triangular = function (Matrix[double] LI)
  return   (Matrix[double] LO)
{
  n = nrow (LI);
  LO = matrix (0, rows = n, cols = n);
  LO = LO + diag (1 / diag (LI));
  
  k = 1;
  while (k < n)
  {
    LPF = matrix (0, rows = n, cols = n);
    parfor (p in 0:((n - 1) / (2 * k)), check = 0)
    {
    i = 2 * k * p;
    j = i + k;
    q = min (n, j + k);
    if (j + 1 <= q) {
      L1 = LO [i + 1:j, i + 1:j];
      L2 = LI [j + 1:q, i + 1:j];
      L3 = LO [j + 1:q, j + 1:q];
      LPF [j + 1:q, i + 1:j] = -L3 %*% L2 %*% L1;
    }
    }
    LO = LO + LPF;
    k = 2 * k;
  }
}

# simple 10x10 test matrix
n = 10;
A = rand (rows = n, cols = n, min = -1, max = 1, pdf = "uniform", sparsity = 1.0)
Mask = cumsum (diag (matrix (1, rows = n, cols = 1)))

# Generate L for stability of the inverse
L = (A %*% t(A)) * Mask

X = invert_lower_triangular (L);
```

#### Cumulative summation with Decay multiplier

Given matrix X, compute:

Y[i] = X[i]
     + X[i-1] * C[i]
     + X[i-2] * C[i] * C[i-1]
     + X[i-3] * C[i] * C[i-1] * C[i-2]
     + ...

```dml
cumsum_prod = function (Matrix[double] X, Matrix[double] C, double start)
  return (Matrix[double] Y)
{
   Y = X;  P = C;  m = nrow(X);  k = 1;
   Y[1,] = Y[1,] + C[1,] * start
   while (k < m) {
     Y[k + 1:m,] = Y[k + 1:m,] + 
                   Y[1:m - k,] * P[k + 1:m,]
     P[k + 1:m,] = P[1:m - k,] * P[k + 1:m,]
     k = 2 * k
   }
}

X = matrix ("1 2 3 4 5 6 7 8 9", rows = 9, cols = 1)

# Zeros in C cause "breaks" that restart the cumulative summation from 0
C = matrix ("0 1 1 0 1 1 1 0 1", rows = 9, cols = 1)

Y = cumsum_prod(X, C, 0)
```

