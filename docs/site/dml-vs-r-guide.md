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

