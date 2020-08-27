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

#### `dml` to `R` translation recipes

To ease the prototyping of the `dml` scripts from its `R` counterparts, this
guide covers various practical functions or operations.

NOTE: This document is still a work in progress.

## Table of Contents

  * [Multiple outputs](#multiple-outputs)
  * [Order function](#order-function)
  * [Read function](#read-function)
  * [`Write` function](#write-function)

##### Multiple-outputs

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

##### `order`-function

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

##### `Read`-function

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

##### `Write`-function

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
