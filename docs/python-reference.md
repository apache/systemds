---
layout: global
title: Reference Guide for Python users
description: Reference Guide for Python users
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

## Introduction

SystemML enables flexible, scalable machine learning. This flexibility is achieved through the specification of a high-level declarative machine learning language that comes in two flavors, 
one with an R-like syntax (DML) and one with a Python-like syntax (PyDML).

Algorithm scripts written in DML and PyDML can be run on Hadoop, on Spark, or in Standalone mode. 
No script modifications are required to change between modes. SystemML automatically performs advanced optimizations 
based on data and cluster characteristics, so much of the need to manually tweak algorithms is largely reduced or eliminated.
To understand more about DML and PyDML, we recommend that you read [Beginner's Guide to DML and PyDML](https://apache.github.io/incubator-systemml/beginners-guide-to-dml-and-pydml.html).

For convenience of Python users, SystemML exposes several language-level APIs that allow Python users to use SystemML
and its algorithms without the need to know DML or PyDML. We explain these APIs in the below sections.

## matrix API

The matrix class allows users to perform linear algebra operations in SystemML using a NumPy-like interface.
This class supports several arithmetic operators (such as +, -, *, /, ^, etc) and also supports most of NumPy's universal functions (i.e. ufuncs).

The current version of NumPy explicitly disables overriding ufunc, but this should be enabled in next release. 
Until then to test above code, please use:

```bash
git clone https://github.com/niketanpansare/numpy.git
cd numpy
python setup.py install
```

This will enable NumPy's functions to invoke matrix class:

```python
import systemml as sml
import numpy as np
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
np.add(m1, m2)
``` 

The matrix class doesnot support following ufuncs:

- Complex number related ufunc (for example: `conj`)
- Hyperbolic/inverse-hyperbolic functions (for example: sinh, arcsinh, cosh, ...)
- Bitwise operators
- Xor operator
- Infinite/Nan-checking (for example: isreal, iscomplex, isfinite, isinf, isnan)
- Other ufuncs: copysign, nextafter, modf, frexp, trunc.

This class also supports several input/output formats such as NumPy arrays, Pandas DataFrame, SciPy sparse matrix and PySpark DataFrame.

By default, the operations are evaluated lazily to avoid conversion overhead and also to maximize optimization scope.
To disable lazy evaluation, please us `set_lazy` method:

```python
>>> import systemml as sml
>>> import numpy as np
>>> m1 = sml.matrix(np.ones((3,3)) + 2)

Welcome to Apache SystemML!

>>> m2 = sml.matrix(np.ones((3,3)) + 3)
>>> np.add(m1, m2) + m1
# This matrix (mVar4) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
mVar2 = load(" ", format="csv")
mVar1 = load(" ", format="csv")
mVar3 = mVar1 + mVar2
mVar4 = mVar3 + mVar1
save(mVar4, " ")


>>> sml.set_lazy(False)
>>> m1 = sml.matrix(np.ones((3,3)) + 2)
>>> m2 = sml.matrix(np.ones((3,3)) + 3)
>>> np.add(m1, m2) + m1
# This matrix (mVar8) is backed by NumPy array. To fetch the NumPy array, invoke toNumPy() method.
``` 

### Usage:

```python
import systemml as sml
import numpy as np
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()
```

Output:

```bash
array([[-60.],
       [-60.],
       [-60.]])
```


### Reference Documentation:

 *class*`systemml.defmatrix.matrix`(*data*, *op=None*)
:   Bases: `object`

    matrix class is a python wrapper that implements basic matrix
    operators, matrix functions as well as converters to common Python
    types (for example: Numpy arrays, PySpark DataFrame and Pandas
    DataFrame).

    The operators supported are:

    1.  Arithmetic operators: +, -, *, /, //, %, \** as well as dot
        (i.e. matrix multiplication)
    2.  Indexing in the matrix
    3.  Relational/Boolean operators: \<, \<=, \>, \>=, ==, !=, &, \|

    In addition, following functions are supported for matrix:

    1.  transpose
    2.  Aggregation functions: sum, mean, var, sd, max, min, argmin,
        argmax, cumsum
    3.  Global statistical built-In functions: exp, log, abs, sqrt,
        round, floor, ceil, sin, cos, tan, asin, acos, atan, sign, solve

    For all the above functions, we always return a two dimensional matrix, especially for aggregation functions with axis. 
    For example: Assuming m1 is a matrix of (3, n), NumPy returns a 1d vector of dimension (3,) for operation m1.sum(axis=1)
    whereas SystemML returns a 2d matrix of dimension (3, 1).
    
    Note: an evaluated matrix contains a data field computed by eval
    method as DataFrame or NumPy array.

        >>> import SystemML as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)

    Welcome to Apache SystemML!

        >>> m1 = sml.matrix(np.ones((3,3)) + 2)
        >>> m2 = sml.matrix(np.ones((3,3)) + 3)
        >>> m2 = m1 * (m2 + m1)
        >>> m4 = 1.0 - m2
        >>> m4
        # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
        mVar1 = load(" ", format="csv")
        mVar2 = load(" ", format="csv")
        mVar3 = mVar2 + mVar1
        mVar4 = mVar1 * mVar3
        mVar5 = 1.0 - mVar4
        save(mVar5, " ")
        >>> m2.eval()
        >>> m2
        # This matrix (mVar4) is backed by NumPy array. To fetch the NumPy array, invoke toNumPy() method.
        >>> m4
        # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
        mVar4 = load(" ", format="csv")
        mVar5 = 1.0 - mVar4
        save(mVar5, " ")
        >>> m4.sum(axis=1).toNumPy()
        array([[-60.],
               [-60.],
               [-60.]])

    Design Decisions:

    1.  Until eval() method is invoked, we create an AST (not exposed to
        the user) that consist of unevaluated operations and data
        required by those operations. As an anology, a spark user can
        treat eval() method similar to calling RDD.persist() followed by
        RDD.count().
    2.  The AST consist of two kinds of nodes: either of type matrix or
        of type DMLOp. Both these classes expose \_visit method, that
        helps in traversing the AST in DFS manner.
    3.  A matrix object can either be evaluated or not. If evaluated,
        the attribute 'data' is set to one of the supported types (for
        example: NumPy array or DataFrame). In this case, the attribute
        'op' is set to None. If not evaluated, the attribute 'op' which
        refers to one of the intermediate node of AST and if of type
        DMLOp. In this case, the attribute 'data' is set to None.

    5.  DMLOp has an attribute 'inputs' which contains list of matrix
        objects or DMLOp.

    6.  To simplify the traversal, every matrix object is considered
        immutable and an matrix operations creates a new matrix object.
        As an example: m1 = sml.matrix(np.ones((3,3))) creates a matrix
        object backed by 'data=(np.ones((3,3))'. m1 = m1 \* 2 will
        create a new matrix object which is now backed by 'op=DMLOp( ...
        )' whose input is earlier created matrix object.

    7.  Left indexing (implemented in \_\_setitem\_\_ method) is a
        special case, where Python expects the existing object to be
        mutated. To ensure the above property, we make deep copy of
        existing object and point any references to the left-indexed
        matrix to the newly created object. Then the left-indexed matrix
        is set to be backed by DMLOp consisting of following pydml:
        left-indexed-matrix = new-deep-copied-matrix
        left-indexed-matrix[index] = value

    8.  Please use m.print\_ast() and/or type m for debugging. Here is a
        sample session:

            >>> npm = np.ones((3,3))
            >>> m1 = sml.matrix(npm + 3)
            >>> m2 = sml.matrix(npm + 5)
            >>> m3 = m1 + m2
            >>> m3
            mVar2 = load(" ", format="csv")
            mVar1 = load(" ", format="csv")
            mVar3 = mVar1 + mVar2
            save(mVar3, " ")
            >>> m3.print_ast()
            - [mVar3] (op).
              - [mVar1] (data).
              - [mVar2] (data).    

 `abs`()
:   

 `acos`()
:   

 `arccos`()
:   

 `arcsin`()
:   

 `arctan`()
:   

 `argmax`(*axis=None*)
:   Returns the indices of the maximum values along an axis.

    axis : int, optional (only axis=1, i.e. rowIndexMax is supported
    in this version)

 `argmin`(*axis=None*)
:   Returns the indices of the minimum values along an axis.

    axis : int, optional (only axis=1, i.e. rowIndexMax is supported
    in this version)

 `asfptype`()
:   

 `asin`()
:   

 `astype`(*t*)
:   

 `atan`()
:   

 `ceil`()
:   

 `cos`()
:   

 `cumsum`(*axis=None*)
:   Returns the indices of the maximum values along an axis.

    axis : int, optional (only axis=0, i.e. cumsum along the rows is
    supported in this version)

 `deg2rad`()
:   Convert angles from degrees to radians.

 `dot`(*other*)[](#systemml.defmatrix.matrix.dot "Permalink to this definition")
:   Numpy way of performing matrix multiplication

 `eval`(*outputDF=False*)[](#systemml.defmatrix.matrix.eval "Permalink to this definition")
:   This is a convenience function that calls the global eval method

 `exp`()[](#systemml.defmatrix.matrix.exp "Permalink to this definition")
:   

 `exp2`()[](#systemml.defmatrix.matrix.exp2 "Permalink to this definition")
:   

 `expm1`()[](#systemml.defmatrix.matrix.expm1 "Permalink to this definition")
:   

 `floor`()[](#systemml.defmatrix.matrix.floor "Permalink to this definition")
:   

 `get_shape`()[](#systemml.defmatrix.matrix.get_shape "Permalink to this definition")
:   

 `ldexp`(*other*)[](#systemml.defmatrix.matrix.ldexp "Permalink to this definition")
:   

 `log`(*y=None*)[](#systemml.defmatrix.matrix.log "Permalink to this definition")
:   

 `log10`()[](#systemml.defmatrix.matrix.log10 "Permalink to this definition")
:   

 `log1p`()[](#systemml.defmatrix.matrix.log1p "Permalink to this definition")
:   

 `log2`()[](#systemml.defmatrix.matrix.log2 "Permalink to this definition")
:   

 `logaddexp`(*other*)[](#systemml.defmatrix.matrix.logaddexp "Permalink to this definition")
:   

 `logaddexp2`(*other*)[](#systemml.defmatrix.matrix.logaddexp2 "Permalink to this definition")
:   

 `logical_not`()[](#systemml.defmatrix.matrix.logical_not "Permalink to this definition")
:   

 `max`(*other=None*, *axis=None*)[](#systemml.defmatrix.matrix.max "Permalink to this definition")
:   Compute the maximum value along the specified axis

    other: matrix or numpy array (& other supported types) or scalar
    axis : int, optional

 `mean`(*axis=None*)[](#systemml.defmatrix.matrix.mean "Permalink to this definition")
:   Compute the arithmetic mean along the specified axis

    axis : int, optional

 `min`(*other=None*, *axis=None*)[](#systemml.defmatrix.matrix.min "Permalink to this definition")
:   Compute the minimum value along the specified axis

    other: matrix or numpy array (& other supported types) or scalar
    axis : int, optional

 `mod`(*other*)[](#systemml.defmatrix.matrix.mod "Permalink to this definition")
:   

 `ndim`*= 2*[](#systemml.defmatrix.matrix.ndim "Permalink to this definition")
:   

 `negative`()[](#systemml.defmatrix.matrix.negative "Permalink to this definition")
:   

 `ones_like`()[](#systemml.defmatrix.matrix.ones_like "Permalink to this definition")
:   

 `print_ast`()[](#systemml.defmatrix.matrix.print_ast "Permalink to this definition")
:   Please use m.print\_ast() and/or type m for debugging. Here is a
    sample session:

        >>> npm = np.ones((3,3))
        >>> m1 = sml.matrix(npm + 3)
        >>> m2 = sml.matrix(npm + 5)
        >>> m3 = m1 + m2
        >>> m3
        mVar2 = load(" ", format="csv")
        mVar1 = load(" ", format="csv")
        mVar3 = mVar1 + mVar2
        save(mVar3, " ")
        >>> m3.print_ast()
        - [mVar3] (op).
          - [mVar1] (data).
          - [mVar2] (data).

 `rad2deg`()[](#systemml.defmatrix.matrix.rad2deg "Permalink to this definition")
:   Convert angles from radians to degrees.

 `reciprocal`()[](#systemml.defmatrix.matrix.reciprocal "Permalink to this definition")
:   

 `remainder`(*other*)[](#systemml.defmatrix.matrix.remainder "Permalink to this definition")
:   

 `round`()[](#systemml.defmatrix.matrix.round "Permalink to this definition")
:   

 `script`*= None*[](#systemml.defmatrix.matrix.script "Permalink to this definition")
:   

 `sd`(*axis=None*)[](#systemml.defmatrix.matrix.sd "Permalink to this definition")
:   Compute the standard deviation along the specified axis

    axis : int, optional

 `set_shape`(*shape*)[](#systemml.defmatrix.matrix.set_shape "Permalink to this definition")
:   

 `shape`[](#systemml.defmatrix.matrix.shape "Permalink to this definition")
:   

 `sign`()[](#systemml.defmatrix.matrix.sign "Permalink to this definition")
:   

 `sin`()[](#systemml.defmatrix.matrix.sin "Permalink to this definition")
:   

 `sqrt`()[](#systemml.defmatrix.matrix.sqrt "Permalink to this definition")
:   

 `square`()[](#systemml.defmatrix.matrix.square "Permalink to this definition")
:   

 `sum`(*axis=None*)[](#systemml.defmatrix.matrix.sum "Permalink to this definition")
:   Compute the sum along the specified axis. 

    axis : int, optional

 `systemmlVarID`*= 0*[](#systemml.defmatrix.matrix.systemmlVarID "Permalink to this definition")
:   

 `tan`()[](#systemml.defmatrix.matrix.tan "Permalink to this definition")
:   

 `toDF`()[](#systemml.defmatrix.matrix.toDF "Permalink to this definition")
:   This is a convenience function that calls the global eval method
    and then converts the matrix object into DataFrame.

 `toNumPy`()[](#systemml.defmatrix.matrix.toNumPy "Permalink to this definition")
:   This is a convenience function that calls the global eval method
    and then converts the matrix object into NumPy array.

 `toPandas`()[](#systemml.defmatrix.matrix.toPandas "Permalink to this definition")
:   This is a convenience function that calls the global eval method
    and then converts the matrix object into Pandas DataFrame.

 `trace`()[](#systemml.defmatrix.matrix.trace "Permalink to this definition")
:   Return the sum of the cells of the main diagonal square matrix

 `transpose`()[](#systemml.defmatrix.matrix.transpose "Permalink to this definition")
:   Transposes the matrix.

 `var`(*axis=None*)[](#systemml.defmatrix.matrix.var "Permalink to this definition")
:   Compute the variance along the specified axis

    axis : int, optional

 `zeros_like`()[](#systemml.defmatrix.matrix.zeros_like "Permalink to this definition")
:   

 `systemml.defmatrix.eval`(*outputs*, *outputDF=False*, *execute=True*)[](#systemml.defmatrix.eval "Permalink to this definition")
:   Executes the unevaluated DML script and computes the matrices
    specified by outputs.

    outputs: list of matrices or a matrix object outputDF: back the data
    of matrix as PySpark DataFrame

 `systemml.defmatrix.solve`(*A*, *b*)[](#systemml.defmatrix.solve "Permalink to this definition")
:   Computes the least squares solution for system of linear equations A
    %\*% x = b

        >>> import numpy as np
        >>> from sklearn import datasets
        >>> import SystemML as sml
        >>> from pyspark.sql import SQLContext
        >>> diabetes = datasets.load_diabetes()
        >>> diabetes_X = diabetes.data[:, np.newaxis, 2]
        >>> X_train = diabetes_X[:-20]
        >>> X_test = diabetes_X[-20:]
        >>> y_train = diabetes.target[:-20]
        >>> y_test = diabetes.target[-20:]
        >>> sml.setSparkContext(sc)
        >>> X = sml.matrix(X_train)
        >>> y = sml.matrix(y_train)
        >>> A = X.transpose().dot(X)
        >>> b = X.transpose().dot(y)
        >>> beta = sml.solve(A, b).toNumPy()
        >>> y_predicted = X_test.dot(beta)
        >>> print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2))
        Residual sum of squares: 25282.12

 `systemml.defmatrix.set_lazy`(*isLazy*)[](#systemml.defmatrix.set_max_depth "Permalink to this definition")
:   This method allows users to set whether the matrix operations should be executed in lazy manner.

    isLazy: True if matrix operations should be evaluated in lazy manner.

 `systemml.defmatrix.debug_array_conversion`(*throwError*)[](#systemml.defmatrix.debug_array_conversion "Permalink to this definition")
:   

 `systemml.random.sampling.normal`(*loc=0.0*, *scale=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.sampling.normal "Permalink to this definition")
:   Draw random samples from a normal (Gaussian) distribution.

    loc: Mean ('centre') of the distribution. scale: Standard deviation
    (spread or 'width') of the distribution. size: Output shape (only
    tuple of length 2, i.e. (m, n), supported). sparsity: Sparsity
    (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.normal(loc=3, scale=2, size=(3,3))
        >>> m1.toNumPy()
        array([[ 3.48857226,  6.17261819,  2.51167259],
               [ 3.60506708, -1.90266305,  3.97601633],
               [ 3.62245706,  5.9430881 ,  2.53070413]])

 `systemml.random.sampling.uniform`(*low=0.0*, *high=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.sampling.uniform "Permalink to this definition")
:   Draw samples from a uniform distribution.

    low: Lower boundary of the output interval. high: Upper boundary of
    the output interval. size: Output shape (only tuple of length 2,
    i.e. (m, n), supported). sparsity: Sparsity (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.uniform(size=(3,3))
        >>> m1.toNumPy()
        array([[ 0.54511396,  0.11937437,  0.72975775],
               [ 0.14135946,  0.01944448,  0.52544478],
               [ 0.67582422,  0.87068849,  0.02766852]])

 `systemml.random.sampling.poisson`(*lam=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.sampling.poisson "Permalink to this definition")
:   Draw samples from a Poisson distribution.

    lam: Expectation of interval, should be \> 0. size: Output shape
    (only tuple of length 2, i.e. (m, n), supported). sparsity: Sparsity
    (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.poisson(lam=1, size=(3,3))
        >>> m1.toNumPy()
        array([[ 1.,  0.,  2.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  0.]])



## MLContext API

The Spark MLContext API offers a programmatic interface for interacting with SystemML from Spark using languages such as Scala, Java, and Python. 
As a result, it offers a convenient way to interact with SystemML from the Spark Shell and from Notebooks such as Jupyter and Zeppelin.

### Usage

The below example demonstrates how to invoke the algorithm [scripts/algorithms/MultiLogReg.dml](https://github.com/apache/incubator-systemml/blob/master/scripts/algorithms/MultiLogReg.dml)
using Python [MLContext API](https://apache.github.io/incubator-systemml/spark-mlcontext-programming-guide).

```python
from sklearn import datasets, neighbors
from pyspark.sql import DataFrame, SQLContext
import systemml as sml
import pandas as pd
import os, imp
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
X_df = sqlCtx.createDataFrame(pd.DataFrame(X_digits[:.9 * n_samples]))
y_df = sqlCtx.createDataFrame(pd.DataFrame(y_digits[:.9 * n_samples]))
ml = sml.MLContext(sc)
# Get the path of MultiLogReg.dml
scriptPath = os.path.join(imp.find_module("systemml")[1], 'systemml-java', 'scripts', 'algorithms', 'MultiLogReg.dml')
script = sml.dml(scriptPath).input(X=X_df, Y_vec=y_df).output("B_out")
beta = ml.execute(script).get('B_out').toNumPy()
```

### Reference documentation

 *class*`systemml.mlcontext.MLResults`(*results*, *sc*)[](#systemml.mlcontext.MLResults "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Wrapper around a Java ML Results object.

    results: JavaObject
    :   A Java MLResults object as returned by calling ml.execute().
    sc: SparkContext
    :   SparkContext

     `get`(*\*outputs*)[](#systemml.mlcontext.MLResults.get "Permalink to this definition")
    :   outputs: string, list of strings
        :   Output variables as defined inside the DML script.

 *class*`systemml.mlcontext.MLContext`(*sc*)[](#systemml.mlcontext.MLContext "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Wrapper around the new SystemML MLContext.

    sc: SparkContext
    :   SparkContext

 `execute`(*script*)[](#systemml.mlcontext.MLContext.execute "Permalink to this definition")
:   Execute a DML / PyDML script.

    script: Script instance
    :   Script instance defined with the appropriate input and
        output variables.

    ml\_results: MLResults
    :   MLResults instance.

 `setExplain`(*explain*)[](#systemml.mlcontext.MLContext.setExplain "Permalink to this definition")
:   Explanation about the program. Mainly intended for developers.

    explain: boolean

 `setExplainLevel`(*explainLevel*)[](#systemml.mlcontext.MLContext.setExplainLevel "Permalink to this definition")
:   Set explain level.

    explainLevel: string
    :   Can be one of 'hops', 'runtime', 'recompile\_hops',
        'recompile\_runtime' or in the above in upper case.

 `setStatistics`(*statistics*)[](#systemml.mlcontext.MLContext.setStatistics "Permalink to this definition")
:   Whether or not to output statistics (such as execution time,
    elapsed time) about script executions.

    statistics: boolean

 `setStatisticsMaxHeavyHitters`(*maxHeavyHitters*)[](#systemml.mlcontext.MLContext.setStatisticsMaxHeavyHitters "Permalink to this definition")
:   The maximum number of heavy hitters that are printed as part of
    the statistics.

    maxHeavyHitters: int

 *class*`systemml.mlcontext.Script`(*scriptString*, *scriptType='dml'*)[](#systemml.mlcontext.Script "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Instance of a DML/PyDML Script.

    scriptString: string
    :   Can be either a file path to a DML script or a DML script
        itself.
    scriptType: string
    :   Script language, either 'dml' for DML (R-like) or 'pydml' for
        PyDML (Python-like).

 `input`(*\*args*, *\*\*kwargs*)[](#systemml.mlcontext.Script.input "Permalink to this definition")
:   args: name, value tuple
    :   where name is a string, and currently supported value
        formats are double, string, dataframe, rdd, and list of such
        object.
    kwargs: dict of name, value pairs
    :   To know what formats are supported for name and value, look
        above.

 `output`(*\*names*)[](#systemml.mlcontext.Script.output "Permalink to this definition")
:   names: string, list of strings
    :   Output variables as defined inside the DML script.

 `systemml.mlcontext.dml`(*scriptString*)[](#systemml.mlcontext.dml "Permalink to this definition")
:   Create a dml script object based on a string.

    scriptString: string
    :   Can be a path to a dml script or a dml script itself.

    script: Script instance
    :   Instance of a script object.

 `systemml.mlcontext.pydml`(*scriptString*)[](#systemml.mlcontext.pydml "Permalink to this definition")
:   Create a pydml script object based on a string.

    scriptString: string
    :   Can be a path to a pydml script or a pydml script itself.

    script: Script instance
    :   Instance of a script object.

 `systemml.mlcontext.getNumCols`(*numPyArr*)[](#systemml.mlcontext.getNumCols "Permalink to this definition")
:   

 `systemml.mlcontext.convertToMatrixBlock`(*sc*, *src*)[](#systemml.mlcontext.convertToMatrixBlock "Permalink to this definition")
:   

 `systemml.mlcontext.convertToNumPyArr`(*sc*, *mb*)[](#systemml.mlcontext.convertToNumPyArr "Permalink to this definition")
:   

 `systemml.mlcontext.convertToPandasDF`(*X*)[](#systemml.mlcontext.convertToPandasDF "Permalink to this definition")
:   

 `systemml.mlcontext.convertToLabeledDF`(*sqlCtx*, *X*, *y=None*)[](#systemml.mlcontext.convertToLabeledDF "Permalink to this definition")
:   


## mllearn API

### Usage

```python
# Scikit-learn way
from sklearn import datasets, neighbors
from systemml.mllearn import LogisticRegression
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target 
n_samples = len(X_digits)
X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]
logistic = LogisticRegression(sqlCtx)
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
```

Output:

```bash
LogisticRegression score: 0.922222
```

### Reference documentation

 *class*`systemml.mllearn.estimators.LinearRegression`(*sqlCtx*, *fit\_intercept=True*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.estimators.LinearRegression "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLRegressor`{.xref .py
    .py-class .docutils .literal}

    Performs linear regression to model the relationship between one
    numerical response variable and one or more explanatory (feature)
    variables.

        >>> import numpy as np
        >>> from sklearn import datasets
        >>> from systemml.mllearn import LinearRegression
        >>> from pyspark.sql import SQLContext
        >>> # Load the diabetes dataset
        >>> diabetes = datasets.load_diabetes()
        >>> # Use only one feature
        >>> diabetes_X = diabetes.data[:, np.newaxis, 2]
        >>> # Split the data into training/testing sets
        >>> diabetes_X_train = diabetes_X[:-20]
        >>> diabetes_X_test = diabetes_X[-20:]
        >>> # Split the targets into training/testing sets
        >>> diabetes_y_train = diabetes.target[:-20]
        >>> diabetes_y_test = diabetes.target[-20:]
        >>> # Create linear regression object
        >>> regr = LinearRegression(sqlCtx, solver='newton-cg')
        >>> # Train the model using the training sets
        >>> regr.fit(diabetes_X_train, diabetes_y_train)
        >>> # The mean square error
        >>> print("Residual sum of squares: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

 *class*`systemml.mllearn.estimators.LogisticRegression`(*sqlCtx*, *penalty='l2'*, *fit\_intercept=True*, *max\_iter=100*, *max\_inner\_iter=0*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.estimators.LogisticRegression "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLClassifier`{.xref
    .py .py-class .docutils .literal}

    Performs both binomial and multinomial logistic regression.

    Scikit-learn way

        >>> from sklearn import datasets, neighbors
        >>> from systemml.mllearn import LogisticRegression
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> digits = datasets.load_digits()
        >>> X_digits = digits.data
        >>> y_digits = digits.target + 1
        >>> n_samples = len(X_digits)
        >>> X_train = X_digits[:.9 * n_samples]
        >>> y_train = y_digits[:.9 * n_samples]
        >>> X_test = X_digits[.9 * n_samples:]
        >>> y_test = y_digits[.9 * n_samples:]
        >>> logistic = LogisticRegression(sqlCtx)
        >>> print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))

    MLPipeline way

        >>> from pyspark.ml import Pipeline
        >>> from systemml.mllearn import LogisticRegression
        >>> from pyspark.ml.feature import HashingTF, Tokenizer
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> training = sqlCtx.createDataFrame([
        >>>     (0L, "a b c d e spark", 1.0),
        >>>     (1L, "b d", 2.0),
        >>>     (2L, "spark f g h", 1.0),
        >>>     (3L, "hadoop mapreduce", 2.0),
        >>>     (4L, "b spark who", 1.0),
        >>>     (5L, "g d a y", 2.0),
        >>>     (6L, "spark fly", 1.0),
        >>>     (7L, "was mapreduce", 2.0),
        >>>     (8L, "e spark program", 1.0),
        >>>     (9L, "a e c l", 2.0),
        >>>     (10L, "spark compile", 1.0),
        >>>     (11L, "hadoop software", 2.0)
        >>> ], ["id", "text", "label"])
        >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
        >>> hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
        >>> lr = LogisticRegression(sqlCtx)
        >>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        >>> model = pipeline.fit(training)
        >>> test = sqlCtx.createDataFrame([
        >>>     (12L, "spark i j k"),
        >>>     (13L, "l m n"),
        >>>     (14L, "mapreduce spark"),
        >>>     (15L, "apache hadoop")], ["id", "text"])
        >>> prediction = model.transform(test)
        >>> prediction.show()

 *class*`systemml.mllearn.estimators.SVM`(*sqlCtx*, *fit\_intercept=True*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *is\_multi\_class=False*, *transferUsingDF=False*)(#systemml.mllearn.estimators.SVM "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLClassifier`{.xref
    .py .py-class .docutils .literal}

    Performs both binary-class and multiclass SVM (Support Vector
    Machines).

        >>> from sklearn import datasets, neighbors
        >>> from systemml.mllearn import SVM
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> digits = datasets.load_digits()
        >>> X_digits = digits.data
        >>> y_digits = digits.target 
        >>> n_samples = len(X_digits)
        >>> X_train = X_digits[:.9 * n_samples]
        >>> y_train = y_digits[:.9 * n_samples]
        >>> X_test = X_digits[.9 * n_samples:]
        >>> y_test = y_digits[.9 * n_samples:]
        >>> svm = SVM(sqlCtx, is_multi_class=True)
        >>> print('LogisticRegression score: %f' % svm.fit(X_train, y_train).score(X_test, y_test))

 *class*`systemml.mllearn.estimators.NaiveBayes`(*sqlCtx*, *laplace=1.0*, *transferUsingDF=False*)(#systemml.mllearn.estimators.NaiveBayes "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLClassifier`{.xref
    .py .py-class .docutils .literal}

    Performs Naive Bayes.

        >>> from sklearn.datasets import fetch_20newsgroups
        >>> from sklearn.feature_extraction.text import TfidfVectorizer
        >>> from systemml.mllearn import NaiveBayes
        >>> from sklearn import metrics
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        >>> newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
        >>> newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
        >>> vectorizer = TfidfVectorizer()
        >>> # Both vectors and vectors_test are SciPy CSR matrix
        >>> vectors = vectorizer.fit_transform(newsgroups_train.data)
        >>> vectors_test = vectorizer.transform(newsgroups_test.data)
        >>> nb = NaiveBayes(sqlCtx)
        >>> nb.fit(vectors, newsgroups_train.target)
        >>> pred = nb.predict(vectors_test)
        >>> metrics.f1_score(newsgroups_test.target, pred, average='weighted')


## Utility classes (used internally)

### systemml.classloader 

 `systemml.classloader.createJavaObject`(*sc*, *obj\_type*)[](#systemml.classloader.createJavaObject "Permalink to this definition")
:   Performs appropriate check if SystemML.jar is available and returns
    the handle to MLContext object on JVM

    sc: SparkContext
    :   SparkContext

    obj\_type: Type of object to create ('mlcontext' or 'dummy')

### systemml.converters

 `systemml.converters.getNumCols`(*numPyArr*)[](#systemml.converters.getNumCols "Permalink to this definition")
:   

 `systemml.converters.convertToMatrixBlock`(*sc*, *src*)[](#systemml.converters.convertToMatrixBlock "Permalink to this definition")
:   

 `systemml.converters.convertToNumPyArr`(*sc*, *mb*)[](#systemml.converters.convertToNumPyArr "Permalink to this definition")
:   

 `systemml.converters.convertToPandasDF`(*X*)[](#systemml.converters.convertToPandasDF "Permalink to this definition")
:   

 `systemml.converters.convertToLabeledDF`(*sqlCtx*, *X*, *y=None*)[](#systemml.converters.convertToLabeledDF "Permalink to this definition")
:  

### Other classes from systemml.defmatrix

 *class*`systemml.defmatrix.DMLOp`(*inputs*, *dml=None*)[](#systemml.defmatrix.DMLOp "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Represents an intermediate node of Abstract syntax tree created to
    generate the PyDML script


## Troubleshooting Python APIs

#### Unable to load SystemML.jar into current pyspark session.

While using SystemML's Python package through pyspark or notebook (SparkContext is not previously created in the session), the
below method is not required. However, if the user wishes to use SystemML through spark-submit and has not previously invoked 

 `systemml.defmatrix.setSparkContext`(*sc*)
:   Before using the matrix, the user needs to invoke this function if SparkContext is not previously created in the session.

    sc: SparkContext
    :   SparkContext

Example:

```python
import systemml as sml
import numpy as np
sml.setSparkContext(sc)
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()
```

If SystemML was not installed via pip, you may have to download SystemML.jar and provide it to pyspark via `--driver-class-path` and `--jars`. 

#### matrix API is running slow when set_lazy(False) or when eval() is called often.

This is a known issue. The matrix API is slow in this scenario due to slow Py4J conversion from Java MatrixObject or Java RDD to Python NumPy or DataFrame.
To resolve this for now, we recommend writing the matrix to FileSystemML and using `load` function.

#### maximum recursion depth exceeded

SystemML matrix is backed by lazy evaluation and uses a recursive Depth First Search (DFS).
Python can throw `RuntimeError: maximum recursion depth exceeded` when the recursion of DFS exceeds beyond the limit 
set by Python. There are two ways to address it:

1. Increase the limit in Python:
 
	```python
	import sys
	some_large_number = 2000
	sys.setrecursionlimit(some_large_number)
	```

2. Evaluate the intermeditate matrix to cut-off large recursion.