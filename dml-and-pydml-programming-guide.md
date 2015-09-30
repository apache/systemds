---
layout: global
title: DML and PyDML Programming Guide
description: DML and PyDML Programming Guide
---

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>


# Overview

SystemML enables *flexible*, scalable machine learning. This flexibility is achieved
through the specification of a high-level declarative machine learning language
that comes in two flavors, one with an R-like syntax (DML) and one with
a Python-like syntax (PyDML).

Algorithm scripts written in DML and PyDML can be run on Hadoop, on Spark, or
in Standalone mode. No script modifications are required to change between modes.
SystemML automatically performs advanced
optimizations based on data and cluster characteristics, so much of the need to manually
tweak algorithms is largely reduced or eliminated.

This SystemML Programming Guide serves as a starting point for writing DML and PyDML 
scripts.


# Script Invocation

DML and PyDML scripts can be invoked in a variety of ways. Suppose that we have `hello.dml` and
`hello.pydml` scripts containing the following:

	print('hello ' + $1)

One way to begin working with SystemML is to build the project and unpack the standalone distribution,
which features the `runStandaloneSystemML.sh` and `runStandaloneSystemML.bat` scripts. The name of the DML or PyDML script
is passed as the first argument to these scripts, along with a variety of arguments.

	./runStandaloneSystemML.sh hello.dml -args world
	./runStandaloneSystemML.sh hello.pydml -python -args world

For DML and PyDML script invocations that take multiple arguments, a common technique is to create
a standard script that invokes `runStandaloneSystemML.sh` or `runStandaloneSystemML.bat` with the arguments specified.

SystemML itself is written in Java and is managed using Maven. As a result, SystemML can readily be
imported into a standard development environment such as Eclipse.
The `DMLScript` class serves as the main entrypoint to SystemML. Executing
`DMLScript` with no arguments displays usage information. A script file can be specified using the `-f` argument.

In Eclipse, a Debug Configuration can be created with `DMLScript` as the Main class and any arguments specified as
Program arguments. A PyDML script requires the addition of a `-python` switch.

<div class="codetabs">

<div data-lang="Eclipse Debug Configuration - Main" markdown="1">
![Eclipse Debug Configuration - Main](img/dml-and-pydml-programming-guide/dmlscript-debug-configuration-hello-world-main-class.png "DMLScript Debug Configuration, Main class")
</div>

<div data-lang="Eclipse Debug Configuration - Arguments" markdown="1">
![Eclipse Debug Configuration - Arguments](img/dml-and-pydml-programming-guide/dmlscript-debug-configuration-hello-world-program-arguments.png "DMLScript Debug Configuration, Program arguments")
</div>

</div>

SystemML contains a default set of configuration information. In addition to this, SystemML looks for a default `./SystemML-config.xml` file in the working directory, where overriding configuration information can be specified. Furthermore, a config file can be specified using the `-config` argument, as in this example:

	-f hello.dml -config=src/main/standalone/SystemML-config.xml -args world

When operating in a distributed environment, it is *highly recommended* that cluster-specific configuration information
is provided to SystemML via a configuration file for optimal performance.


# Data Types

SystemML has four value data types. In DML, these are: **double**, **integer**,
**string**, and **boolean**. In PyDML, these are: **float**, **int**,
**str**, and **bool**. In normal usage, the data type of a variable is implicit
based on its value. Mathematical operations typically operate on
doubles/floats, whereas integers/ints are typically useful for tasks such as
iteration and accessing elements in a matrix.

<div class="codetabs">

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
*Note that matrices index from 1 in both DML and PyDML.*

<div class="codetabs">

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
for (i in 1:nrow(m)):
    for (j in 1:ncol(m)):
        n = m[i,j]
        print('[' + i + ',' + j + ']:' + scalar(n))
{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
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

</div>

For additional information about the **`matrix()`** and **`full()`** functions, please see the 
DML Language Reference ([Matrix Construction](dml-language-reference.html#matrix-construction-manipulation-and-aggregation-built-in-functions)) and the 
PyDML Language Reference (Matrix Construction).


## Saving a Matrix

A matrix can be saved using the **`write()`** function in DML and the **`save()`** function in PyDML. SystemML supports four
different formats: **`text`** (`i,j,v`), **`mm`** (`Matrix Market`), **`csv`** (`delimiter-separated values`), and **`binary`**.

<div class="codetabs">

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
metadata within the *.mm file. All formats are text-based except binary. The contents of the resulting files are shown here.

<div class="codetabs">

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
metadata file is required, except for the Matrix Market format.

<div class="codetabs">

<div data-lang="DML" markdown="1">
{% highlight r %}
m = read("m.txt")
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
m = load("m.txt")
print("min:" + min(m))
print("max:" + max(m))
print("sum:" + sum(m))
mRowSums = rowSums(m)
for (i in 1:nrow(mRowSums)):
    print("row " + i + " sum:" + scalar(mRowSums[i,1]))
mColSums = colSums(m)
for (i in 1:ncol(mColSums)):
    print("col " + i + " sum:" + scalar(mColSums[1,i]))
{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
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

</div>


## Matrix Operations

DML and PyDML offer a rich set of operators and built-in functions to perform various operations on matrices and scalars.
Operators and built-in functions are described in great detail in the DML Language Reference 
([Expressions](dml-language-reference.html#expressions), [Built-In Functions](dml-language-reference.html#built-in-functions))
and the PyDML Language Reference
(Expressions, Built-In Functions).

In this example, we create a matrix A. Next, we create another matrix B by adding 4 to each element in A. Next, we flip
B by taking its transpose. We then multiply A and B, represented by matrix C. We create a matrix D with the same number
of rows and columns as C, and initialize its elements to 5. We then subtract D from C and divide the values of its elements
by 2 and assign the resulting matrix to D.

This example also shows a user-defined function called `printMatrix()`, which takes a string and matrix as arguments and returns
nothing.

<div class="codetabs">

<div data-lang="DML" markdown="1">
{% highlight r %}
printMatrix = function(string which, matrix[double] mat) {
    print(which)
    for (i in 1:nrow(mat)) {
        colVals = '| '
        for (j in 1:ncol(mat)) {
            n = mat[i,j]
            colVals = colVals + as.scalar(n) + ' | '
        }
        print(colVals)
    }
}

A = matrix("1 2 3 4 5 6", rows=3, cols=2)
z = printMatrix('Matrix A:', A)
B = A + 4
B = t(B)
z = printMatrix('Matrix B:', B)
C = A %*% B
z = printMatrix('Matrix C:', C)
D = matrix(5, rows=nrow(C), cols=ncol(C))
D = (C - D) / 2
z = printMatrix('Matrix D:', D)

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
def printMatrix(which: str, mat: matrix[float]):
    print(which)
    for (i in 1:nrow(mat)):
        colVals = '| '
        for (j in 1:ncol(mat)):
            n = mat[i,j]
            colVals = colVals + scalar(n) + ' | '
        print(colVals)

A = full("1 2 3 4 5 6", rows=3, cols=2)
z = printMatrix('Matrix A:', A)
B = A + 4
B = transpose(B)
z = printMatrix('Matrix B:', B)
C = dot(A, B)
z = printMatrix('Matrix C:', C)
D = full(5, rows=nrow(C), cols=ncol(C))
D = (C - D) / 2
z = printMatrix('Matrix D:', D)

{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
	Matrix A:
	| 1.0 | 2.0 | 
	| 3.0 | 4.0 | 
	| 5.0 | 6.0 | 
	Matrix B:
	| 5.0 | 7.0 | 9.0 | 
	| 6.0 | 8.0 | 10.0 | 
	Matrix C:
	| 17.0 | 23.0 | 29.0 | 
	| 39.0 | 53.0 | 67.0 | 
	| 61.0 | 83.0 | 105.0 | 
	Matrix D:
	| 6.0 | 9.0 | 12.0 | 
	| 17.0 | 24.0 | 31.0 | 
	| 28.0 | 39.0 | 50.0 | 
</div>

</div>


## Matrix Indexing

The elements in a matrix can be accessed by their row and column indices. In the example below, we have 3x3 matrix A.
First, we access the element at the third row and third column. Next, we obtain a row slice (vector) of the matrix by
specifying row 2 and leaving the column blank. We obtain a column slice (vector) by leaving the row blank and specifying
column 3. After that, we obtain a submatrix via range indexing, where we specify rows 2 to 3, separated by a colon, and columns
1 to 2, separated by a colon.

<div class="codetabs">

<div data-lang="DML" markdown="1">
{% highlight r %}
printMatrix = function(string which, matrix[double] mat) {
    print(which)
    for (i in 1:nrow(mat)) {
        colVals = '| '
        for (j in 1:ncol(mat)) {
            n = mat[i,j]
            colVals = colVals + as.scalar(n) + ' | '
        }
        print(colVals)
    }
}

A = matrix("1 2 3 4 5 6 7 8 9", rows=3, cols=3)
z = printMatrix('Matrix A:', A)
B = A[3,3]
z = printMatrix('Matrix B:', B)
C = A[2,]
z = printMatrix('Matrix C:', C)
D = A[,3]
z = printMatrix('Matrix D:', D)
E = A[2:3,1:2]
z = printMatrix('Matrix E:', E)

{% endhighlight %}
</div>

<div data-lang="PyDML" markdown="1">
{% highlight python %}
def printMatrix(which: str, mat: matrix[float]):
    print(which)
    for (i in 1:nrow(mat)):
        colVals = '| '
        for (j in 1:ncol(mat)):
            n = mat[i,j]
            colVals = colVals + scalar(n) + ' | '
        print(colVals)

A = full("1 2 3 4 5 6 7 8 9", rows=3, cols=3)
z = printMatrix('Matrix A:', A)
B = A[3,3]
z = printMatrix('Matrix B:', B)
C = A[2,]
z = printMatrix('Matrix C:', C)
D = A[,3]
z = printMatrix('Matrix D:', D)
E = A[2:3,1:2]
z = printMatrix('Matrix E:', E)

{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
	Matrix A:
	| 1.0 | 2.0 | 3.0 | 
	| 4.0 | 5.0 | 6.0 | 
	| 7.0 | 8.0 | 9.0 | 
	Matrix B:
	| 9.0 | 
	Matrix C:
	| 4.0 | 5.0 | 6.0 | 
	Matrix D:
	| 3.0 | 
	| 6.0 | 
	| 9.0 | 
	Matrix E:
	| 4.0 | 5.0 | 
	| 7.0 | 8.0 | 
</div>

</div>


# Control Statements

DML and PyDML both feature `if` and `if-else` conditional statements. In addition, DML features `else-if` which avoids the
need for nested conditional statements.

DML and PyDML feature 3 loop statements: `while`, `for`, and `parfor` (parallel for). In the example, note that the 
`print` statements within the `parfor` loop can occur in any order since the iterations occur in parallel rather than
sequentially as in a regular `for` loop. The `parfor` statement can include several optional parameters, as described
in the DML Language Reference ([ParFor Statement](dml-language-reference.html#parfor-statement)) and PyDML Language Reference (ParFor Statement).

<div class="codetabs">

<div data-lang="DML" markdown="1">
{% highlight r %}
i = 1
while (i < 3) {
    if (i == 1) {
        print('hello')
    } else {
        print('world')
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
while (i < 3):
    if (i == 1):
        print('hello')
    else:
        print('world')
    i = i + 1

A = full("1 2 3 4 5 6", rows=3, cols=2)

for (i in 1:nrow(A)):
    print("for A[" + i + ",1]:" + scalar(A[i,1]))

parfor(i in 1:nrow(A)):
    print("parfor A[" + i + ",1]:" + scalar(A[i,1]))
{% endhighlight %}
</div>

<div data-lang="Result" markdown="1">
	hello
	world
	for A[1,1]:1.0
	for A[2,1]:3.0
	for A[3,1]:5.0
	parfor A[2,1]:3.0
	parfor A[1,1]:1.0
	parfor A[3,1]:5.0
</div>

</div>


# User-Defined Functions

Functions encapsulate useful functionality in SystemML. In addition to built-in functions, users can define their own functions.
Functions take 0 or more parameters and return 0 or more values.
Currently, if a function returns nothing, it still needs to be assigned to a variable.

<div class="codetabs">

<div data-lang="DML" markdown="1">
{% highlight r %}
doSomething = function(matrix[double] mat) return (matrix[double] ret) {
    additionalCol = matrix(1, rows=nrow(mat), cols=1) # 1x3 matrix with 1 values
    ret = append(mat, additionalCol) # append column to matrix
    ret = append(ret, seq(0, 2, 1))  # append column (0,1,2) to matrix
    ret = append(ret, rowMaxs(ret))  # append column of max row values to matrix
    ret = append(ret, rowSums(ret))  # append column of row sums to matrix
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
    ret = append(mat, additionalCol) # append column to matrix
    ret = append(ret, seq(0, 2, 1))  # append column (0,1,2) to matrix
    ret = append(ret, rowMaxs(ret))  # append column of max row values to matrix
    ret = append(ret, rowSums(ret))  # append column of row sums to matrix

A = rand(rows=3, cols=2, min=0, max=2) # random 3x2 matrix with values 0 to 2
B = doSomething(A)
save(A, "A.csv", format="csv")
save(B, "B.csv", format="csv")
{% endhighlight %}
</div>

</div>

In the above example, a 3x2 matrix of random doubles between 0 and 2 is created using the **`rand()`** function.
Additional parameters can be passed to **`rand()`** to control sparsity and other matrix characteristics.

Matrix A is passed to the `doSomething` function. A column of 1 values is appended to the matrix. A column
consisting of the values `(0, 1, 2)` is appended to the matrix. Next, a column consisting of the maximum row values
is appended to the matrix. A column consisting of the row sums is appended to the matrix, and this resulting
matrix is returned to variable B. Matrix A is output to the `A.csv` file and matrix B is saved as the `B.csv` file.


<div class="codetabs">

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

<div class="codetabs">

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

for (i in 1:numRowsToPrint):
    for (j in 1:numColsToPrint):
        print('[' + i + ',' + j + ']:' + scalar(m[i,j]))

{% endhighlight %}
</div>

<div data-lang="DML Named Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.dml -nvargs M=M.txt rowsToPrint=1 colsToPrint=3
	
	Example #1 Results:
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	
	Example #2 Arguments:
	-f ex.dml -nvargs M=M.txt
	
	Example #2 Results:
	[1,1]:1.0
	[1,2]:2.0
	[2,1]:0.0
	[2,2]:0.0
	
</div>

<div data-lang="PyDML Named Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.pydml -python -nvargs M=M.txt rowsToPrint=1 colsToPrint=3
	
	Example #1 Results:
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	
	Example #2 Arguments:
	-f ex.pydml -python -nvargs M=M.txt
	
	Example #2 Results:
	[1,1]:1.0
	[1,2]:2.0
	[2,1]:0.0
	[2,2]:0.0
	
</div>

</div>

Here, we see identical functionality but with positional arguments.

<div class="codetabs">

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

for (i in 1:numRowsToPrint):
    for (j in 1:numColsToPrint):
        print('[' + i + ',' + j + ']:' + scalar(m[i,j]))

{% endhighlight %}
</div>

<div data-lang="DML Positional Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.dml -args M.txt 1 3
	
	Example #1 Results:
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	
	Example #2 Arguments:
	-f ex.dml -args M.txt
	
	Example #2 Results:
	[1,1]:1.0
	[1,2]:2.0
	[2,1]:0.0
	[2,2]:0.0
	
</div>

<div data-lang="PyDML Positional Arguments and Results" markdown="1">
	Example #1 Arguments:
	-f ex.pydml -python -args M.txt 1 3
	
	Example #1 Results:
	[1,1]:1.0
	[1,2]:2.0
	[1,3]:3.0
	
	Example #2 Arguments:
	-f ex.pydml -python -args M.txt
	
	Example #2 Results:
	[1,1]:1.0
	[1,2]:2.0
	[2,1]:0.0
	[2,2]:0.0
	
</div>

</div>


# Additional Information

The [DML Language Reference](dml-language-reference.html) and PyDML Language Reference contain highly detailed information regard DML 
and PyDML.

In addition, many excellent examples of DML and PyDML can be found in the `system-ml/scripts` and 
`system-ml/test/scripts/applications` directories.

