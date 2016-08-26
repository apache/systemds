---
layout: global
title: Beginner's Guide to DML and PyDML
description: Beginner's Guide to DML and PyDML
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

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>


# Overview

SystemML enables *flexible*, scalable machine learning. This flexibility is achieved
through the specification of a high-level declarative machine learning language
that comes in two flavors, one with an R-like syntax (DML) and one with
a Python-like syntax (PyDML).

Algorithm scripts written in DML and PyDML can be run on Spark, on Hadoop, or
in Standalone mode. SystemML also features an MLContext API that allows SystemML
to be accessed via Scala or Python from a Spark Shell, a Jupyter Notebook, or a Zeppelin Notebook.

This Beginner's Guide serves as a starting point for writing DML and PyDML
scripts.


# Script Invocation

DML and PyDML scripts can be invoked in a variety of ways. Suppose that we have `hello.dml` and
`hello.pydml` scripts containing the following:

	print('hello ' + $1)

One way to begin working with SystemML is to [download a standalone tar.gz or zip distribution of SystemML](http://systemml.apache.org/download.html)
and use the `runStandaloneSystemML.sh` and `runStandaloneSystemML.bat` scripts to run SystemML in standalone
mode. The name of the DML or PyDML script
is passed as the first argument to these scripts, along with a variety of arguments.

	./runStandaloneSystemML.sh hello.dml -args world
	./runStandaloneSystemML.sh hello.pydml -python -args world


# Data Types

SystemML has four value data types. In DML, these are: **double**, **integer**,
**string**, and **boolean**. In PyDML, these are: **float**, **int**,
**str**, and **bool**. In normal usage, the data type of a variable is implicit
based on its value. Mathematical operations typically operate on
doubles/floats, whereas integers/ints are typically useful for tasks such as
iteration and accessing elements in a matrix.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
aDouble = 3.0
bInteger = 2
print('aDouble = ' + aDouble)
print('bInteger = ' + bInteger)
print('aDouble + bInteger = ' + (aDouble + bInteger))
print('bInteger ^ 3 = ' + (bInteger ^ 3))
print('aDouble ^ 2 = ' + (aDouble ^ 2))

cBoolean = TRUE
print('cBoolean = ' + cBoolean)
print('(2 < 1) = ' + (2 < 1))

dString = 'Hello'
eString = dString + ' World'
print('dString = ' + dString)
print('eString = ' + eString)
{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
aFloat = 3.0
bInt = 2
print('aFloat = ' + aFloat)
print('bInt = ' + bInt)
print('aFloat + bInt = ' + (aFloat + bInt))
print('bInt ** 3 = ' + (bInt ** 3))
print('aFloat ** 2 = ' + (aFloat ** 2))

cBool = True
print('cBool = ' + cBool)
print('(2 < 1) = ' + (2 < 1))

dStr = 'Hello'
eStr = dStr + ' World'
print('dStr = ' + dStr)
print('eStr = ' + eStr)
{% endhighlight %}
</div>

<div data-lang="DML Result" markdown="1">
	aDouble = 3.0
	bInteger = 2
	aDouble + bInteger = 5.0
	bInteger ^ 3 = 8.0
	aDouble ^ 2 = 9.0
	cBoolean = TRUE
	(2 < 1) = FALSE
	dString = Hello
	eString = Hello World
</div>

<div data-lang="PyDML Result" markdown="1">
	aFloat = 3.0
	bInt = 2
	aFloat + bInt = 5.0
	bInt ** 3 = 8.0
	aFloat ** 2 = 9.0
	cBool = TRUE
	(2 < 1) = FALSE
	dStr = Hello
	eStr = Hello World
</div>

</div>


# Matrix Basics

## Creating a Matrix

A matrix can be created in DML using the **`matrix()`** function and in PyDML using the **`full()`** 
function. In the example below, a matrix element is still considered to be of the matrix data type,
so the value is cast to a scalar in order to print it. Matrix element values are of type **double**/**float**.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
m = matrix("1 2 3 4 5 6 7 8 9 10 11 12", rows=4, cols=3)
for (i in 1:nrow(m)) {
    for (j in 1:ncol(m)) {
        n = m[i,j]
        print('[' + i + ',' + j + ']:' + as.scalar(n))
    }
}
{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
m = full("1 2 3 4 5 6 7 8 9 10 11 12", rows=4, cols=3)
for (i in 0:nrow(m)-1):
    for (j in 0:ncol(m)-1):
        n = m[i,j]
        print('[' + i + ',' + j + ']:' + scalar(n))
{% endhighlight %}
</div>

<div data-lang="DML Result" markdown="1">
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	[2,1]:4.0
	[2,2]:5.0
	[2,3]:6.0
	[3,1]:7.0
	[3,2]:8.0
	[3,3]:9.0
	[4,1]:10.0
	[4,2]:11.0
	[4,3]:12.0
</div>

<div data-lang="PyDML Result" markdown="1">
	[0,0]:1.0
	[0,1]:2.0
	[0,2]:3.0
	[1,0]:4.0
	[1,1]:5.0
	[1,2]:6.0
	[2,0]:7.0
	[2,1]:8.0
	[2,2]:9.0
	[3,0]:10.0
	[3,1]:11.0
	[3,2]:12.0
</div>

</div>

We can also output the matrix element values using the **`toString`** function:

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
m = matrix("1 2 3 4 5 6 7 8 9 10 11 12", rows=4, cols=3)
print(toString(m, sep=" | ", decimal=1))
{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
m = full("1 2 3 4 5 6 7 8 9 10 11 12", rows=4, cols=3)
print(toString(m, sep=" | ", decimal=1))
{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
	1.0 | 2.0 | 3.0
	4.0 | 5.0 | 6.0
	7.0 | 8.0 | 9.0
	10.0 | 11.0 | 12.0
</div>

</div>


For additional information about the **`matrix()`** and **`full()`** functions, please see the 
[Matrix Construction](dml-language-reference.html#matrix-construction-manipulation-and-aggregation-built-in-functions)
section of the Language Reference. For information about the **`toString()`** function, see
the [Other Built-In Functions](dml-language-reference.html#other-built-in-functions) section of the Language Reference.


## Saving a Matrix

A matrix can be saved using the **`write()`** function in DML and the **`save()`** function in PyDML. SystemML supports four
different formats: **`text`** (`i,j,v`), **`mm`** (`Matrix Market`), **`csv`** (`delimiter-separated values`), and **`binary`**.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
m = matrix("1 2 3 0 0 0 7 8 9 0 0 0", rows=4, cols=3)
write(m, "m.txt", format="text")
write(m, "m.mm", format="mm")
write(m, "m.csv", format="csv")
write(m, "m.binary", format="binary")
{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
m = full("1 2 3 0 0 0 7 8 9 0 0 0", rows=4, cols=3)
save(m, "m.txt", format="text")
save(m, "m.mm", format="mm")
save(m, "m.csv", format="csv")
save(m, "m.binary", format="binary")
{% endhighlight %}
</div>

</div>

Saving a matrix automatically creates a metadata file for each format except for Matrix Market, since Matrix Market contains
metadata within the \*.mm file. All formats are text-based except binary. The contents of the resulting files are shown here.
*Note that the **`text`** (`i,j,v`) and **`mm`** (`Matrix Market`) formats index from 1, even when working with PyDML, which
is 0-based.*

<div class="codetabs2">

<div data-lang="m.txt" markdown="1">
	1 1 1.0
	1 2 2.0
	1 3 3.0
	3 1 7.0
	3 2 8.0
	3 3 9.0
</div>

<div data-lang="m.txt.mtd" markdown="1">
	{ 
	    "data_type": "matrix"
	    ,"value_type": "double"
	    ,"rows": 4
	    ,"cols": 3
	    ,"nnz": 6
	    ,"format": "text"
	    ,"description": { "author": "SystemML" } 
	}
</div>

<div data-lang="m.mm" markdown="1">
	%%MatrixMarket matrix coordinate real general
	4 3 6
	1 1 1.0
	1 2 2.0
	1 3 3.0
	3 1 7.0
	3 2 8.0
	3 3 9.0
</div>

<div data-lang="m.csv" markdown="1">
	1.0,2.0,3.0
	0,0,0
	7.0,8.0,9.0
	0,0,0
</div>

<div data-lang="m.csv.mtd" markdown="1">
	{ 
	    "data_type": "matrix"
	    ,"value_type": "double"
	    ,"rows": 4
	    ,"cols": 3
	    ,"nnz": 6
	    ,"format": "csv"
	    ,"header": false
	    ,"sep": ","
	    ,"description": { "author": "SystemML" } 
	}
</div>

<div data-lang="m.binary" markdown="1">
	Not text-based
</div>

<div data-lang="m.binary.mtd" markdown="1">
	{ 
	    "data_type": "matrix"
	    ,"value_type": "double"
	    ,"rows": 4
	    ,"cols": 3
	    ,"rows_in_block": 1000
	    ,"cols_in_block": 1000
	    ,"nnz": 6
	    ,"format": "binary"
	    ,"description": { "author": "SystemML" } 
	}
</div>

</div>


## Loading a Matrix

A matrix can be loaded using the **`read()`** function in DML and the **`load()`** function in PyDML. As with saving, SystemML supports four
formats: **`text`** (`i,j,v`), **`mm`** (`Matrix Market`), **`csv`** (`delimiter-separated values`), and **`binary`**. To read a file, a corresponding
metadata file is required, except for the Matrix Market format. A metadata file is not required if a `format` parameter is specified to the **`read()`**
or **`load()`** functions.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
m = read("m.csv")
print("min:" + min(m))
print("max:" + max(m))
print("sum:" + sum(m))
mRowSums = rowSums(m)
for (i in 1:nrow(mRowSums)) {
    print("row " + i + " sum:" + as.scalar(mRowSums[i,1]))
}
mColSums = colSums(m)
for (i in 1:ncol(mColSums)) {
    print("col " + i + " sum:" + as.scalar(mColSums[1,i]))
}
{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
m = load("m.csv")
print("min:" + min(m))
print("max:" + max(m))
print("sum:" + sum(m))
mRowSums = rowSums(m)
for (i in 0:nrow(mRowSums)-1):
    print("row " + i + " sum:" + scalar(mRowSums[i,0]))
mColSums = colSums(m)
for (i in 0:ncol(mColSums)-1):
    print("col " + i + " sum:" + scalar(mColSums[0,i]))
    
{% endhighlight %}
</div>

<div data-lang="DML Result" markdown="1">
	min:0.0
	max:9.0
	sum:30.0
	row 1 sum:6.0
	row 2 sum:0.0
	row 3 sum:24.0
	row 4 sum:0.0
	col 1 sum:8.0
	col 2 sum:10.0
	col 3 sum:12.0
</div>

<div data-lang="PyDML Result" markdown="1">
	min:0.0
	max:9.0
	sum:30.0
	row 0 sum:6.0
	row 1 sum:0.0
	row 2 sum:24.0
	row 3 sum:0.0
	col 0 sum:8.0
	col 1 sum:10.0
	col 2 sum:12.0
</div>

</div>


## Matrix Operations

DML and PyDML offer a rich set of operators and built-in functions to perform various operations on matrices and scalars.
Operators and built-in functions are described in great detail in the Language Reference
([Expressions](dml-language-reference.html#expressions), [Built-In Functions](dml-language-reference.html#built-in-functions)).

In this example, we create a matrix A. Next, we create another matrix B by adding 4 to each element in A. Next, we flip
B by taking its transpose. We then multiply A and B, represented by matrix C. We create a matrix D with the same number
of rows and columns as C, and initialize its elements to 5. We then subtract D from C and divide the values of its elements
by 2 and assign the resulting matrix to D.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
A = matrix("1 2 3 4 5 6", rows=3, cols=2)
print(toString(A))
B = A + 4
B = t(B)
print(toString(B))
C = A %*% B
print(toString(C))
D = matrix(5, rows=nrow(C), cols=ncol(C))
D = (C - D) / 2
print(toString(D))

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
A = full("1 2 3 4 5 6", rows=3, cols=2)
print(toString(A))
B = A + 4
B = transpose(B)
print(toString(B))
C = dot(A, B)
print(toString(C))
D = full(5, rows=nrow(C), cols=ncol(C))
D = (C - D) / 2
print(toString(D))

{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
	1.000 2.000
	3.000 4.000
	5.000 6.000
	
	5.000 7.000 9.000
	6.000 8.000 10.000
	
	17.000 23.000 29.000
	39.000 53.000 67.000
	61.000 83.000 105.000
	
	6.000 9.000 12.000
	17.000 24.000 31.000
	28.000 39.000 50.000
</div>

</div>


## Matrix Indexing

The elements in a matrix can be accessed by their row and column indices. In the example below, we have 3x3 matrix A.
First, we access the element at the third row and third column. Next, we obtain a row slice (vector) of the matrix by
specifying the row and leaving the column blank. We obtain a column slice (vector) by leaving the row blank and specifying
the column. After that, we obtain a submatrix via range indexing, where we specify rows, separated by a colon, and columns,
separated by a colon.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
A = matrix("1 2 3 4 5 6 7 8 9", rows=3, cols=3)
print(toString(A))
B = A[3,3]
print(toString(B))
C = A[2,]
print(toString(C))
D = A[,3]
print(toString(D))
E = A[2:3,1:2]
print(toString(E))

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
A = full("1 2 3 4 5 6 7 8 9", rows=3, cols=3)
print(toString(A))
B = A[2,2]
print(toString(B))
C = A[1,]
print(toString(C))
D = A[,2]
print(toString(D))
E = A[1:3,0:2]
print(toString(E))

{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
	1.000 2.000 3.000
	4.000 5.000 6.000
	7.000 8.000 9.000
	
	9.000
	
	4.000 5.000 6.000
	
	3.000
	6.000
	9.000
	
	4.000 5.000
	7.000 8.000
	
</div>

</div>


# Control Statements

DML and PyDML both feature `if`, `if-else`, and `if-else-if` conditional statements.

DML and PyDML feature 3 loop statements: `while`, `for`, and `parfor` (parallel for). In the example, note that the 
`print` statements within the `parfor` loop can occur in any order since the iterations occur in parallel rather than
sequentially as in a regular `for` loop. The `parfor` statement can include several optional parameters, as described
in the Language Reference ([ParFor Statement](dml-language-reference.html#parfor-statement)).

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
i = 1
while (i <= 3) {
    if (i == 1) {
        print('hello')
    } else if (i == 2) {
        print('world')
    } else {
        print('!!!')
    }
    i = i + 1
}

A = matrix("1 2 3 4 5 6", rows=3, cols=2)

for (i in 1:nrow(A)) {
    print("for A[" + i + ",1]:" + as.scalar(A[i,1]))
}

parfor(i in 1:nrow(A)) {
    print("parfor A[" + i + ",1]:" + as.scalar(A[i,1]))
}

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
i = 1
while (i <= 3):
    if (i == 1):
        print('hello')
    elif (i == 2):
        print('world')
    else:
        print('!!!')
    i = i + 1

A = full("1 2 3 4 5 6", rows=3, cols=2)

for (i in 0:nrow(A)-1):
    print("for A[" + i + ",0]:" + scalar(A[i,0]))

parfor(i in 0:nrow(A)-1):
    print("parfor A[" + i + ",0]:" + scalar(A[i,0]))

{% endhighlight %}
</div>

<div data-lang="DML Result" markdown="1">
	hello
	world
	!!!
	for A[1,1]:1.0
	for A[2,1]:3.0
	for A[3,1]:5.0
	parfor A[2,1]:3.0
	parfor A[1,1]:1.0
	parfor A[3,1]:5.0
</div>

<div data-lang="PyDML Result" markdown="1">
	hello
	world
	!!!
	for A[0,0]:1.0
	for A[1,0]:3.0
	for A[2,0]:5.0
	parfor A[0,0]:1.0
	parfor A[2,0]:5.0
	parfor A[1,0]:3.0
</div>

</div>


# User-Defined Functions

Functions encapsulate useful functionality in SystemML. In addition to built-in functions, users can define their own functions.
Functions take 0 or more parameters and return 0 or more values.
Currently, if a function returns nothing, it still needs to be assigned to a variable.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}
doSomething = function(matrix[double] mat) return (matrix[double] ret) {
    additionalCol = matrix(1, rows=nrow(mat), cols=1) # 1x3 matrix with 1 values
    ret = cbind(mat, additionalCol) # concatenate column to matrix
    ret = cbind(ret, seq(0, 2, 1))  # concatenate column (0,1,2) to matrix
    ret = cbind(ret, rowMaxs(ret))  # concatenate column of max row values to matrix
    ret = cbind(ret, rowSums(ret))  # concatenate column of row sums to matrix
}

A = rand(rows=3, cols=2, min=0, max=2) # random 3x2 matrix with values 0 to 2
B = doSomething(A)
write(A, "A.csv", format="csv")
write(B, "B.csv", format="csv")
{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
def doSomething(mat: matrix[float]) -> (ret: matrix[float]):
    additionalCol = full(1, rows=nrow(mat), cols=1) # 1x3 matrix with 1 values
    ret = cbind(mat, additionalCol) # concatenate column to matrix
    ret = cbind(ret, seq(0, 2, 1))  # concatenate column (0,1,2) to matrix
    ret = cbind(ret, rowMaxs(ret))  # concatenate column of max row values to matrix
    ret = cbind(ret, rowSums(ret))  # concatenate column of row sums to matrix

A = rand(rows=3, cols=2, min=0, max=2) # random 3x2 matrix with values 0 to 2
B = doSomething(A)
save(A, "A.csv", format="csv")
save(B, "B.csv", format="csv")
{% endhighlight %}
</div>

</div>

In the above example, a 3x2 matrix of random doubles between 0 and 2 is created using the **`rand()`** function.
Additional parameters can be passed to **`rand()`** to control sparsity and other matrix characteristics.

Matrix A is passed to the `doSomething` function. A column of 1 values is concatenated to the matrix. A column
consisting of the values `(0, 1, 2)` is concatenated to the matrix. Next, a column consisting of the maximum row values
is concatenated to the matrix. A column consisting of the row sums is concatenated to the matrix, and this resulting
matrix is returned to variable B. Matrix A is output to the `A.csv` file and matrix B is saved as the `B.csv` file.


<div class="codetabs2">

<div data-lang="A.csv" markdown="1">
	1.6091961493071,0.7088614208099939
	0.5984862383600267,1.5732118950764993
	0.2947607068519842,1.9081406573366781
</div>

<div data-lang="B.csv" markdown="1">
	1.6091961493071,0.7088614208099939,1.0,0,1.6091961493071,4.927253719424194
	0.5984862383600267,1.5732118950764993,1.0,1.0,1.5732118950764993,5.744910028513026
	0.2947607068519842,1.9081406573366781,1.0,2.0,2.0,7.202901364188662
</div>

</div>


# Command-Line Arguments and Default Values

Command-line arguments can be passed to DML and PyDML scripts either as named arguments or as positional arguments. Named
arguments are the preferred technique. Named arguments can be passed utilizing the `-nvargs` switch, and positional arguments
can be passed using the `-args` switch.

Default values can be set using the **`ifdef()`** function.

In the example below, a matrix is read from the file system using named argument `M`. The number of rows to print is specified
using the `rowsToPrint` argument, which defaults to 2 if no argument is supplied. Likewise, the number of columns is
specified using `colsToPrint` with a default value of 2.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}

fileM = $M

numRowsToPrint = ifdef($rowsToPrint, 2) # default to 2
numColsToPrint = ifdef($colsToPrint, 2) # default to 2

m = read(fileM)

for (i in 1:numRowsToPrint) {
    for (j in 1:numColsToPrint) {
        print('[' + i + ',' + j + ']:' + as.scalar(m[i,j]))
    }
}

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}

fileM = $M

numRowsToPrint = ifdef($rowsToPrint, 2) # default to 2
numColsToPrint = ifdef($colsToPrint, 2) # default to 2

m = load(fileM)

for (i in 0:numRowsToPrint-1):
    for (j in 0:numColsToPrint-1):
        print('[' + i + ',' + j + ']:' + scalar(m[i,j]))

{% endhighlight %}
</div>

<div data-lang="DML Named Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.dml -nvargs M=m.csv rowsToPrint=1 colsToPrint=3
	
	Example #1 Results:
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	
	Example #2 Arguments:
	-f ex.dml -nvargs M=m.csv
	
	Example #2 Results:
	[1,1]:1.0
	[1,2]:2.0
	[2,1]:0.0
	[2,2]:0.0
	
</div>

<div data-lang="PyDML Named Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.pydml -python -nvargs M=m.csv rowsToPrint=1 colsToPrint=3
	
	Example #1 Results:
	[0,0]:1.0
	[0,1]:2.0
	[0,2]:3.0
	
	Example #2 Arguments:
	-f ex.pydml -python -nvargs M=m.csv
	
	Example #2 Results:
	[0,0]:1.0
	[0,1]:2.0
	[1,0]:0.0
	[1,1]:0.0
	
</div>

</div>

Here, we see identical functionality but with positional arguments.

<div class="codetabs2">

<div data-lang="DML" markdown="1">
{% highlight r %}

fileM = $1

numRowsToPrint = ifdef($2, 2) # default to 2
numColsToPrint = ifdef($3, 2) # default to 2

m = read(fileM)

for (i in 1:numRowsToPrint) {
    for (j in 1:numColsToPrint) {
        print('[' + i + ',' + j + ']:' + as.scalar(m[i,j]))
    }
}

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}

fileM = $1

numRowsToPrint = ifdef($2, 2) # default to 2
numColsToPrint = ifdef($3, 2) # default to 2

m = load(fileM)

for (i in 0:numRowsToPrint-1):
    for (j in 0:numColsToPrint-1):
        print('[' + i + ',' + j + ']:' + scalar(m[i,j]))

{% endhighlight %}
</div>

<div data-lang="DML Positional Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.dml -args m.csv 1 3
	
	Example #1 Results:
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	
	Example #2 Arguments:
	-f ex.dml -args m.csv
	
	Example #2 Results:
	[1,1]:1.0
	[1,2]:2.0
	[2,1]:0.0
	[2,2]:0.0
	
</div>

<div data-lang="PyDML Positional Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.pydml -python -args m.csv 1 3
	
	Example #1 Results:
	[0,0]:1.0
	[0,1]:2.0
	[0,2]:3.0
	
	Example #2 Arguments:
	-f ex.pydml -python -args m.csv
	
	Example #2 Results:
	[0,0]:1.0
	[0,1]:2.0
	[1,0]:0.0
	[1,1]:0.0
	
</div>

</div>


# Additional Information

The [Language Reference](dml-language-reference.html) contains highly detailed information regarding DML.

In addition, many excellent examples of DML and PyDML can be found in the [`scripts`](https://github.com/apache/incubator-systemml/tree/master/scripts) directory.

