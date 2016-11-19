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



systemml package[](#systemml-package "Permalink to this headline")
===================================================================

Subpackages[](#subpackages "Permalink to this headline")
---------------------------------------------------------

-   [systemml.mllearn package](systemml.mllearn.html)
    -   [Submodules](systemml.mllearn.html#submodules)
    -   [systemml.mllearn.estimators
        module](systemml.mllearn.html#module-systemml.mllearn.estimators)
    -   [Module contents](systemml.mllearn.html#module-systemml.mllearn)
        -   [SystemML
            Algorithms](systemml.mllearn.html#systemml-algorithms)
-   [systemml.random package](systemml.random.html)
    -   [Submodules](systemml.random.html#submodules)
    -   [systemml.random.sampling
        module](systemml.random.html#module-systemml.random.sampling)
    -   [Module contents](systemml.random.html#module-systemml.random)
        -   [Random Number
            Generation](systemml.random.html#random-number-generation)

Submodules[](#submodules "Permalink to this headline")
-------------------------------------------------------

systemml.classloader module[](#module-systemml.classloader "Permalink to this headline")
-----------------------------------------------------------------------------------------

 `systemml.classloader.`{.descclassname}`createJavaObject`{.descname}(*sc*, *obj\_type*)[](#systemml.classloader.createJavaObject "Permalink to this definition")
:   Performs appropriate check if SystemML.jar is available and returns
    the handle to MLContext object on JVM

    sc: SparkContext
    :   SparkContext

    obj\_type: Type of object to create (â€˜mlcontextâ€™ or â€˜dummyâ€™)

systemml.converters module[](#module-systemml.converters "Permalink to this headline")
---------------------------------------------------------------------------------------

 `systemml.converters.`{.descclassname}`getNumCols`{.descname}(*numPyArr*)[](#systemml.converters.getNumCols "Permalink to this definition")
:   

 `systemml.converters.`{.descclassname}`convertToMatrixBlock`{.descname}(*sc*, *src*)[](#systemml.converters.convertToMatrixBlock "Permalink to this definition")
:   

 `systemml.converters.`{.descclassname}`convertToNumPyArr`{.descname}(*sc*, *mb*)[](#systemml.converters.convertToNumPyArr "Permalink to this definition")
:   

 `systemml.converters.`{.descclassname}`convertToPandasDF`{.descname}(*X*)[](#systemml.converters.convertToPandasDF "Permalink to this definition")
:   

 `systemml.converters.`{.descclassname}`convertToLabeledDF`{.descname}(*sqlCtx*, *X*, *y=None*)[](#systemml.converters.convertToLabeledDF "Permalink to this definition")
:   

systemml.defmatrix module[](#module-systemml.defmatrix "Permalink to this headline")
-------------------------------------------------------------------------------------

 `systemml.defmatrix.`{.descclassname}`setSparkContext`{.descname}(*sc*)[](#systemml.defmatrix.setSparkContext "Permalink to this definition")
:   Before using the matrix, the user needs to invoke this function.

    sc: SparkContext
    :   SparkContext

 *class*`systemml.defmatrix.`{.descclassname}`matrix`{.descname}(*data*, *op=None*)[](#systemml.defmatrix.matrix "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    matrix class is a python wrapper that implements basic matrix
    operators, matrix functions as well as converters to common Python
    types (for example: Numpy arrays, PySpark DataFrame and Pandas
    DataFrame).

    The operators supported are:

    1.  Arithmetic operators: +, -, *, /, //, %, \** as well as dot
        (i.e. matrix multiplication)
    2.  Indexing in the matrix
    3.  Relational/Boolean operators: \<, \<=, \>, \>=, ==, !=, &, |

    In addition, following functions are supported for matrix:

    1.  transpose
    2.  Aggregation functions: sum, mean, var, sd, max, min, argmin,
        argmax, cumsum
    3.  Global statistical built-In functions: exp, log, abs, sqrt,
        round, floor, ceil, sin, cos, tan, asin, acos, atan, sign, solve

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
        # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.
        mVar1 = load(" ", format="csv")
        mVar2 = load(" ", format="csv")
        mVar3 = mVar2 + mVar1
        mVar4 = mVar1 * mVar3
        mVar5 = 1.0 - mVar4
        save(mVar5, " ")
        >>> m2.eval()
        >>> m2
        # This matrix (mVar4) is backed by NumPy array. To fetch the NumPy array, invoke toNumPyArray() method.
        >>> m4
        # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.
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
        the attribute â€˜dataâ€™ is set to one of the supported types (for
        example: NumPy array or DataFrame). In this case, the attribute
        â€˜opâ€™ is set to None. If not evaluated, the attribute â€˜opâ€™ which
        refers to one of the intermediate node of AST and if of type
        DMLOp. In this case, the attribute â€˜dataâ€™ is set to None.

    5.  DMLOp has an attribute â€˜inputsâ€™ which contains list of matrix
        objects or DMLOp.

    6.  To simplify the traversal, every matrix object is considered
        immutable and an matrix operations creates a new matrix object.
        As an example: m1 = sml.matrix(np.ones((3,3))) creates a matrix
        object backed by â€˜data=(np.ones((3,3))â€™. m1 = m1 \* 2 will
        create a new matrix object which is now backed by â€˜op=DMLOp( ...
        )â€™ whose input is earlier created matrix object.

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

     `THROW_ARRAY_CONVERSION_ERROR`{.descname}*= False*[](#systemml.defmatrix.matrix.THROW_ARRAY_CONVERSION_ERROR "Permalink to this definition")
    :   

     `abs`{.descname}()[](#systemml.defmatrix.matrix.abs "Permalink to this definition")
    :   

     `acos`{.descname}()[](#systemml.defmatrix.matrix.acos "Permalink to this definition")
    :   

     `arccos`{.descname}()[](#systemml.defmatrix.matrix.arccos "Permalink to this definition")
    :   

     `arcsin`{.descname}()[](#systemml.defmatrix.matrix.arcsin "Permalink to this definition")
    :   

     `arctan`{.descname}()[](#systemml.defmatrix.matrix.arctan "Permalink to this definition")
    :   

     `argmax`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.argmax "Permalink to this definition")
    :   Returns the indices of the maximum values along an axis.

        axis : int, optional (only axis=1, i.e. rowIndexMax is supported
        in this version)

     `argmin`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.argmin "Permalink to this definition")
    :   Returns the indices of the minimum values along an axis.

        axis : int, optional (only axis=1, i.e. rowIndexMax is supported
        in this version)

     `asfptype`{.descname}()[](#systemml.defmatrix.matrix.asfptype "Permalink to this definition")
    :   

     `asin`{.descname}()[](#systemml.defmatrix.matrix.asin "Permalink to this definition")
    :   

     `astype`{.descname}(*t*)[](#systemml.defmatrix.matrix.astype "Permalink to this definition")
    :   

     `atan`{.descname}()[](#systemml.defmatrix.matrix.atan "Permalink to this definition")
    :   

     `ceil`{.descname}()[](#systemml.defmatrix.matrix.ceil "Permalink to this definition")
    :   

     `cos`{.descname}()[](#systemml.defmatrix.matrix.cos "Permalink to this definition")
    :   

     `cumsum`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.cumsum "Permalink to this definition")
    :   Returns the indices of the maximum values along an axis.

        axis : int, optional (only axis=0, i.e. cumsum along the rows is
        supported in this version)

     `deg2rad`{.descname}()[](#systemml.defmatrix.matrix.deg2rad "Permalink to this definition")
    :   Convert angles from degrees to radians.

     `dml`{.descname}*= []*[](#systemml.defmatrix.matrix.dml "Permalink to this definition")
    :   

     `dot`{.descname}(*other*)[](#systemml.defmatrix.matrix.dot "Permalink to this definition")
    :   Numpy way of performing matrix multiplication

     `eval`{.descname}(*outputDF=False*)[](#systemml.defmatrix.matrix.eval "Permalink to this definition")
    :   This is a convenience function that calls the global eval method

     `exp`{.descname}()[](#systemml.defmatrix.matrix.exp "Permalink to this definition")
    :   

     `exp2`{.descname}()[](#systemml.defmatrix.matrix.exp2 "Permalink to this definition")
    :   

     `expm1`{.descname}()[](#systemml.defmatrix.matrix.expm1 "Permalink to this definition")
    :   

     `floor`{.descname}()[](#systemml.defmatrix.matrix.floor "Permalink to this definition")
    :   

     `get_shape`{.descname}()[](#systemml.defmatrix.matrix.get_shape "Permalink to this definition")
    :   

     `ldexp`{.descname}(*other*)[](#systemml.defmatrix.matrix.ldexp "Permalink to this definition")
    :   

     `log`{.descname}(*y=None*)[](#systemml.defmatrix.matrix.log "Permalink to this definition")
    :   

     `log10`{.descname}()[](#systemml.defmatrix.matrix.log10 "Permalink to this definition")
    :   

     `log1p`{.descname}()[](#systemml.defmatrix.matrix.log1p "Permalink to this definition")
    :   

     `log2`{.descname}()[](#systemml.defmatrix.matrix.log2 "Permalink to this definition")
    :   

     `logaddexp`{.descname}(*other*)[](#systemml.defmatrix.matrix.logaddexp "Permalink to this definition")
    :   

     `logaddexp2`{.descname}(*other*)[](#systemml.defmatrix.matrix.logaddexp2 "Permalink to this definition")
    :   

     `logical_not`{.descname}()[](#systemml.defmatrix.matrix.logical_not "Permalink to this definition")
    :   

     `max`{.descname}(*other=None*, *axis=None*)[](#systemml.defmatrix.matrix.max "Permalink to this definition")
    :   Compute the maximum value along the specified axis

        other: matrix or numpy array (& other supported types) or scalar
        axis : int, optional

     `mean`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.mean "Permalink to this definition")
    :   Compute the arithmetic mean along the specified axis

        axis : int, optional

     `min`{.descname}(*other=None*, *axis=None*)[](#systemml.defmatrix.matrix.min "Permalink to this definition")
    :   Compute the minimum value along the specified axis

        other: matrix or numpy array (& other supported types) or scalar
        axis : int, optional

     `ml`{.descname}*= None*[](#systemml.defmatrix.matrix.ml "Permalink to this definition")
    :   

     `mod`{.descname}(*other*)[](#systemml.defmatrix.matrix.mod "Permalink to this definition")
    :   

     `ndim`{.descname}*= 2*[](#systemml.defmatrix.matrix.ndim "Permalink to this definition")
    :   

     `negative`{.descname}()[](#systemml.defmatrix.matrix.negative "Permalink to this definition")
    :   

     `ones_like`{.descname}()[](#systemml.defmatrix.matrix.ones_like "Permalink to this definition")
    :   

     `print_ast`{.descname}(*numSpaces=0*)[](#systemml.defmatrix.matrix.print_ast "Permalink to this definition")
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

     `rad2deg`{.descname}()[](#systemml.defmatrix.matrix.rad2deg "Permalink to this definition")
    :   Convert angles from radians to degrees.

     `reciprocal`{.descname}()[](#systemml.defmatrix.matrix.reciprocal "Permalink to this definition")
    :   

     `remainder`{.descname}(*other*)[](#systemml.defmatrix.matrix.remainder "Permalink to this definition")
    :   

     `round`{.descname}()[](#systemml.defmatrix.matrix.round "Permalink to this definition")
    :   

     `script`{.descname}*= None*[](#systemml.defmatrix.matrix.script "Permalink to this definition")
    :   

     `sd`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.sd "Permalink to this definition")
    :   Compute the standard deviation along the specified axis

        axis : int, optional

     `set_shape`{.descname}(*shape*)[](#systemml.defmatrix.matrix.set_shape "Permalink to this definition")
    :   

     `shape`{.descname}[](#systemml.defmatrix.matrix.shape "Permalink to this definition")
    :   

     `sign`{.descname}()[](#systemml.defmatrix.matrix.sign "Permalink to this definition")
    :   

     `sin`{.descname}()[](#systemml.defmatrix.matrix.sin "Permalink to this definition")
    :   

     `sqrt`{.descname}()[](#systemml.defmatrix.matrix.sqrt "Permalink to this definition")
    :   

     `square`{.descname}()[](#systemml.defmatrix.matrix.square "Permalink to this definition")
    :   

     `sum`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.sum "Permalink to this definition")
    :   Compute the sum along the specified axis

        axis : int, optional

     `systemmlVarID`{.descname}*= 0*[](#systemml.defmatrix.matrix.systemmlVarID "Permalink to this definition")
    :   

     `tan`{.descname}()[](#systemml.defmatrix.matrix.tan "Permalink to this definition")
    :   

     `toDataFrame`{.descname}()[](#systemml.defmatrix.matrix.toDataFrame "Permalink to this definition")
    :   This is a convenience function that calls the global eval method
        and then converts the matrix object into DataFrame.

     `toNumPyArray`{.descname}()[](#systemml.defmatrix.matrix.toNumPyArray "Permalink to this definition")
    :   This is a convenience function that calls the global eval method
        and then converts the matrix object into NumPy array.

     `toPandas`{.descname}()[](#systemml.defmatrix.matrix.toPandas "Permalink to this definition")
    :   This is a convenience function that calls the global eval method
        and then converts the matrix object into Pandas DataFrame.

     `trace`{.descname}()[](#systemml.defmatrix.matrix.trace "Permalink to this definition")
    :   Return the sum of the cells of the main diagonal square matrix

     `transpose`{.descname}()[](#systemml.defmatrix.matrix.transpose "Permalink to this definition")
    :   Transposes the matrix.

     `var`{.descname}(*axis=None*)[](#systemml.defmatrix.matrix.var "Permalink to this definition")
    :   Compute the variance along the specified axis

        axis : int, optional

     `visited`{.descname}*= []*[](#systemml.defmatrix.matrix.visited "Permalink to this definition")
    :   

     `zeros_like`{.descname}()[](#systemml.defmatrix.matrix.zeros_like "Permalink to this definition")
    :   

 `systemml.defmatrix.`{.descclassname}`eval`{.descname}(*outputs*, *outputDF=False*, *execute=True*)[](#systemml.defmatrix.eval "Permalink to this definition")
:   Executes the unevaluated DML script and computes the matrices
    specified by outputs.

    outputs: list of matrices or a matrix object outputDF: back the data
    of matrix as PySpark DataFrame

 `systemml.defmatrix.`{.descclassname}`solve`{.descname}(*A*, *b*)[](#systemml.defmatrix.solve "Permalink to this definition")
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

 *class*`systemml.defmatrix.`{.descclassname}`DMLOp`{.descname}(*inputs*, *dml=None*)[](#systemml.defmatrix.DMLOp "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Represents an intermediate node of Abstract syntax tree created to
    generate the PyDML script

     `MAX_DEPTH`{.descname}*= 0*[](#systemml.defmatrix.DMLOp.MAX_DEPTH "Permalink to this definition")
    :   

     `print_ast`{.descname}(*numSpaces*)[](#systemml.defmatrix.DMLOp.print_ast "Permalink to this definition")
    :   

 `systemml.defmatrix.`{.descclassname}`set_max_depth`{.descname}(*depth*)[](#systemml.defmatrix.set_max_depth "Permalink to this definition")
:   This method allows users to set the depth of lazy SystemML DAG.
    Setting this to 1 ensures that every operation is evaluated (which
    can turn off many optimization). Setting this to 0 ensures that
    SystemML will never implicitly evaluate the DAG, but will require
    user to call eval or toPandas/toNumPyArray/... explicitly.

    depth: Should be greater than equal to 0.

 `systemml.defmatrix.`{.descclassname}`debug_array_conversion`{.descname}(*throwError*)[](#systemml.defmatrix.debug_array_conversion "Permalink to this definition")
:   

systemml.mlcontext module[](#systemml-mlcontext-module "Permalink to this headline")
-------------------------------------------------------------------------------------

 *class*`systemml.mlcontext.`{.descclassname}`MLResults`{.descname}(*results*, *sc*)[](#systemml.mlcontext.MLResults "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Wrapper around a Java ML Results object.

    results: JavaObject
    :   A Java MLResults object as returned by calling ml.execute().
    sc: SparkContext
    :   SparkContext

     `get`{.descname}(*\*outputs*)[](#systemml.mlcontext.MLResults.get "Permalink to this definition")
    :   outputs: string, list of strings
        :   Output variables as defined inside the DML script.

 *class*`systemml.mlcontext.`{.descclassname}`MLContext`{.descname}(*sc*)[](#systemml.mlcontext.MLContext "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Wrapper around the new SystemML MLContext.

    sc: SparkContext
    :   SparkContext

     `execute`{.descname}(*script*)[](#systemml.mlcontext.MLContext.execute "Permalink to this definition")
    :   Execute a DML / PyDML script.

        script: Script instance
        :   Script instance defined with the appropriate input and
            output variables.

        ml\_results: MLResults
        :   MLResults instance.

     `setExplain`{.descname}(*explain*)[](#systemml.mlcontext.MLContext.setExplain "Permalink to this definition")
    :   Explanation about the program. Mainly intended for developers.

        explain: boolean

     `setExplainLevel`{.descname}(*explainLevel*)[](#systemml.mlcontext.MLContext.setExplainLevel "Permalink to this definition")
    :   Set explain level.

        explainLevel: string
        :   Can be one of â€œhopsâ€, â€œruntimeâ€, â€œrecompile\_hopsâ€,
            â€œrecompile\_runtimeâ€ or in the above in upper case.

     `setStatistics`{.descname}(*statistics*)[](#systemml.mlcontext.MLContext.setStatistics "Permalink to this definition")
    :   Whether or not to output statistics (such as execution time,
        elapsed time) about script executions.

        statistics: boolean

     `setStatisticsMaxHeavyHitters`{.descname}(*maxHeavyHitters*)[](#systemml.mlcontext.MLContext.setStatisticsMaxHeavyHitters "Permalink to this definition")
    :   The maximum number of heavy hitters that are printed as part of
        the statistics.

        maxHeavyHitters: int

 *class*`systemml.mlcontext.`{.descclassname}`Script`{.descname}(*scriptString*, *scriptType='dml'*)[](#systemml.mlcontext.Script "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Instance of a DML/PyDML Script.

    scriptString: string
    :   Can be either a file path to a DML script or a DML script
        itself.
    scriptType: string
    :   Script language, either â€œdmlâ€ for DML (R-like) or â€œpydmlâ€ for
        PyDML (Python-like).

     `input`{.descname}(*\*args*, *\*\*kwargs*)[](#systemml.mlcontext.Script.input "Permalink to this definition")
    :   args: name, value tuple
        :   where name is a string, and currently supported value
            formats are double, string, dataframe, rdd, and list of such
            object.
        kwargs: dict of name, value pairs
        :   To know what formats are supported for name and value, look
            above.

     `output`{.descname}(*\*names*)[](#systemml.mlcontext.Script.output "Permalink to this definition")
    :   names: string, list of strings
        :   Output variables as defined inside the DML script.

 `systemml.mlcontext.`{.descclassname}`dml`{.descname}(*scriptString*)[](#systemml.mlcontext.dml "Permalink to this definition")
:   Create a dml script object based on a string.

    scriptString: string
    :   Can be a path to a dml script or a dml script itself.

    script: Script instance
    :   Instance of a script object.

 `systemml.mlcontext.`{.descclassname}`pydml`{.descname}(*scriptString*)[](#systemml.mlcontext.pydml "Permalink to this definition")
:   Create a pydml script object based on a string.

    scriptString: string
    :   Can be a path to a pydml script or a pydml script itself.

    script: Script instance
    :   Instance of a script object.

 `systemml.mlcontext.`{.descclassname}`getNumCols`{.descname}(*numPyArr*)[](#systemml.mlcontext.getNumCols "Permalink to this definition")
:   

 `systemml.mlcontext.`{.descclassname}`convertToMatrixBlock`{.descname}(*sc*, *src*)[](#systemml.mlcontext.convertToMatrixBlock "Permalink to this definition")
:   

 `systemml.mlcontext.`{.descclassname}`convertToNumPyArr`{.descname}(*sc*, *mb*)[](#systemml.mlcontext.convertToNumPyArr "Permalink to this definition")
:   

 `systemml.mlcontext.`{.descclassname}`convertToPandasDF`{.descname}(*X*)[](#systemml.mlcontext.convertToPandasDF "Permalink to this definition")
:   

 `systemml.mlcontext.`{.descclassname}`convertToLabeledDF`{.descname}(*sqlCtx*, *X*, *y=None*)[](#systemml.mlcontext.convertToLabeledDF "Permalink to this definition")
:   

Module contents[](#module-systemml "Permalink to this headline")
-----------------------------------------------------------------

 *class*`systemml.`{.descclassname}`MLResults`{.descname}(*results*, *sc*)[](#systemml.MLResults "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Wrapper around a Java ML Results object.

    results: JavaObject
    :   A Java MLResults object as returned by calling ml.execute().
    sc: SparkContext
    :   SparkContext

     `get`{.descname}(*\*outputs*)[](#systemml.MLResults.get "Permalink to this definition")
    :   outputs: string, list of strings
        :   Output variables as defined inside the DML script.

 *class*`systemml.`{.descclassname}`MLContext`{.descname}(*sc*)[](#systemml.MLContext "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Wrapper around the new SystemML MLContext.

    sc: SparkContext
    :   SparkContext

     `execute`{.descname}(*script*)[](#systemml.MLContext.execute "Permalink to this definition")
    :   Execute a DML / PyDML script.

        script: Script instance
        :   Script instance defined with the appropriate input and
            output variables.

        ml\_results: MLResults
        :   MLResults instance.

     `setExplain`{.descname}(*explain*)[](#systemml.MLContext.setExplain "Permalink to this definition")
    :   Explanation about the program. Mainly intended for developers.

        explain: boolean

     `setExplainLevel`{.descname}(*explainLevel*)[](#systemml.MLContext.setExplainLevel "Permalink to this definition")
    :   Set explain level.

        explainLevel: string
        :   Can be one of â€œhopsâ€, â€œruntimeâ€, â€œrecompile\_hopsâ€,
            â€œrecompile\_runtimeâ€ or in the above in upper case.

     `setStatistics`{.descname}(*statistics*)[](#systemml.MLContext.setStatistics "Permalink to this definition")
    :   Whether or not to output statistics (such as execution time,
        elapsed time) about script executions.

        statistics: boolean

     `setStatisticsMaxHeavyHitters`{.descname}(*maxHeavyHitters*)[](#systemml.MLContext.setStatisticsMaxHeavyHitters "Permalink to this definition")
    :   The maximum number of heavy hitters that are printed as part of
        the statistics.

        maxHeavyHitters: int

 *class*`systemml.`{.descclassname}`Script`{.descname}(*scriptString*, *scriptType='dml'*)[](#systemml.Script "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Instance of a DML/PyDML Script.

    scriptString: string
    :   Can be either a file path to a DML script or a DML script
        itself.
    scriptType: string
    :   Script language, either â€œdmlâ€ for DML (R-like) or â€œpydmlâ€ for
        PyDML (Python-like).

     `input`{.descname}(*\*args*, *\*\*kwargs*)[](#systemml.Script.input "Permalink to this definition")
    :   args: name, value tuple
        :   where name is a string, and currently supported value
            formats are double, string, dataframe, rdd, and list of such
            object.
        kwargs: dict of name, value pairs
        :   To know what formats are supported for name and value, look
            above.

     `output`{.descname}(*\*names*)[](#systemml.Script.output "Permalink to this definition")
    :   names: string, list of strings
        :   Output variables as defined inside the DML script.

 `systemml.`{.descclassname}`dml`{.descname}(*scriptString*)[](#systemml.dml "Permalink to this definition")
:   Create a dml script object based on a string.

    scriptString: string
    :   Can be a path to a dml script or a dml script itself.

    script: Script instance
    :   Instance of a script object.

 `systemml.`{.descclassname}`pydml`{.descname}(*scriptString*)[](#systemml.pydml "Permalink to this definition")
:   Create a pydml script object based on a string.

    scriptString: string
    :   Can be a path to a pydml script or a pydml script itself.

    script: Script instance
    :   Instance of a script object.

 `systemml.`{.descclassname}`setSparkContext`{.descname}(*sc*)[](#systemml.setSparkContext "Permalink to this definition")
:   Before using the matrix, the user needs to invoke this function.

    sc: SparkContext
    :   SparkContext

 *class*`systemml.`{.descclassname}`matrix`{.descname}(*data*, *op=None*)[](#systemml.matrix "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    matrix class is a python wrapper that implements basic matrix
    operators, matrix functions as well as converters to common Python
    types (for example: Numpy arrays, PySpark DataFrame and Pandas
    DataFrame).

    The operators supported are:

    1.  Arithmetic operators: +, -, *, /, //, %, \** as well as dot
        (i.e. matrix multiplication)
    2.  Indexing in the matrix
    3.  Relational/Boolean operators: \<, \<=, \>, \>=, ==, !=, &, |

    In addition, following functions are supported for matrix:

    1.  transpose
    2.  Aggregation functions: sum, mean, var, sd, max, min, argmin,
        argmax, cumsum
    3.  Global statistical built-In functions: exp, log, abs, sqrt,
        round, floor, ceil, sin, cos, tan, asin, acos, atan, sign, solve

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
        # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.
        mVar1 = load(" ", format="csv")
        mVar2 = load(" ", format="csv")
        mVar3 = mVar2 + mVar1
        mVar4 = mVar1 * mVar3
        mVar5 = 1.0 - mVar4
        save(mVar5, " ")
        >>> m2.eval()
        >>> m2
        # This matrix (mVar4) is backed by NumPy array. To fetch the NumPy array, invoke toNumPyArray() method.
        >>> m4
        # This matrix (mVar5) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPyArray() or toDataFrame() or toPandas() methods.
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
        the attribute â€˜dataâ€™ is set to one of the supported types (for
        example: NumPy array or DataFrame). In this case, the attribute
        â€˜opâ€™ is set to None. If not evaluated, the attribute â€˜opâ€™ which
        refers to one of the intermediate node of AST and if of type
        DMLOp. In this case, the attribute â€˜dataâ€™ is set to None.

    5.  DMLOp has an attribute â€˜inputsâ€™ which contains list of matrix
        objects or DMLOp.

    6.  To simplify the traversal, every matrix object is considered
        immutable and an matrix operations creates a new matrix object.
        As an example: m1 = sml.matrix(np.ones((3,3))) creates a matrix
        object backed by â€˜data=(np.ones((3,3))â€™. m1 = m1 \* 2 will
        create a new matrix object which is now backed by â€˜op=DMLOp( ...
        )â€™ whose input is earlier created matrix object.

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

     `THROW_ARRAY_CONVERSION_ERROR`{.descname}*= False*[](#systemml.matrix.THROW_ARRAY_CONVERSION_ERROR "Permalink to this definition")
    :   

     `abs`{.descname}()[](#systemml.matrix.abs "Permalink to this definition")
    :   

     `acos`{.descname}()[](#systemml.matrix.acos "Permalink to this definition")
    :   

     `arccos`{.descname}()[](#systemml.matrix.arccos "Permalink to this definition")
    :   

     `arcsin`{.descname}()[](#systemml.matrix.arcsin "Permalink to this definition")
    :   

     `arctan`{.descname}()[](#systemml.matrix.arctan "Permalink to this definition")
    :   

     `argmax`{.descname}(*axis=None*)[](#systemml.matrix.argmax "Permalink to this definition")
    :   Returns the indices of the maximum values along an axis.

        axis : int, optional (only axis=1, i.e. rowIndexMax is supported
        in this version)

     `argmin`{.descname}(*axis=None*)[](#systemml.matrix.argmin "Permalink to this definition")
    :   Returns the indices of the minimum values along an axis.

        axis : int, optional (only axis=1, i.e. rowIndexMax is supported
        in this version)

     `asfptype`{.descname}()[](#systemml.matrix.asfptype "Permalink to this definition")
    :   

     `asin`{.descname}()[](#systemml.matrix.asin "Permalink to this definition")
    :   

     `astype`{.descname}(*t*)[](#systemml.matrix.astype "Permalink to this definition")
    :   

     `atan`{.descname}()[](#systemml.matrix.atan "Permalink to this definition")
    :   

     `ceil`{.descname}()[](#systemml.matrix.ceil "Permalink to this definition")
    :   

     `cos`{.descname}()[](#systemml.matrix.cos "Permalink to this definition")
    :   

     `cumsum`{.descname}(*axis=None*)[](#systemml.matrix.cumsum "Permalink to this definition")
    :   Returns the indices of the maximum values along an axis.

        axis : int, optional (only axis=0, i.e. cumsum along the rows is
        supported in this version)

     `deg2rad`{.descname}()[](#systemml.matrix.deg2rad "Permalink to this definition")
    :   Convert angles from degrees to radians.

     `dml`{.descname}*= []*[](#systemml.matrix.dml "Permalink to this definition")
    :   

     `dot`{.descname}(*other*)[](#systemml.matrix.dot "Permalink to this definition")
    :   Numpy way of performing matrix multiplication

     `eval`{.descname}(*outputDF=False*)[](#systemml.matrix.eval "Permalink to this definition")
    :   This is a convenience function that calls the global eval method

     `exp`{.descname}()[](#systemml.matrix.exp "Permalink to this definition")
    :   

     `exp2`{.descname}()[](#systemml.matrix.exp2 "Permalink to this definition")
    :   

     `expm1`{.descname}()[](#systemml.matrix.expm1 "Permalink to this definition")
    :   

     `floor`{.descname}()[](#systemml.matrix.floor "Permalink to this definition")
    :   

     `get_shape`{.descname}()[](#systemml.matrix.get_shape "Permalink to this definition")
    :   

     `ldexp`{.descname}(*other*)[](#systemml.matrix.ldexp "Permalink to this definition")
    :   

     `log`{.descname}(*y=None*)[](#systemml.matrix.log "Permalink to this definition")
    :   

     `log10`{.descname}()[](#systemml.matrix.log10 "Permalink to this definition")
    :   

     `log1p`{.descname}()[](#systemml.matrix.log1p "Permalink to this definition")
    :   

     `log2`{.descname}()[](#systemml.matrix.log2 "Permalink to this definition")
    :   

     `logaddexp`{.descname}(*other*)[](#systemml.matrix.logaddexp "Permalink to this definition")
    :   

     `logaddexp2`{.descname}(*other*)[](#systemml.matrix.logaddexp2 "Permalink to this definition")
    :   

     `logical_not`{.descname}()[](#systemml.matrix.logical_not "Permalink to this definition")
    :   

     `max`{.descname}(*other=None*, *axis=None*)[](#systemml.matrix.max "Permalink to this definition")
    :   Compute the maximum value along the specified axis

        other: matrix or numpy array (& other supported types) or scalar
        axis : int, optional

     `mean`{.descname}(*axis=None*)[](#systemml.matrix.mean "Permalink to this definition")
    :   Compute the arithmetic mean along the specified axis

        axis : int, optional

     `min`{.descname}(*other=None*, *axis=None*)[](#systemml.matrix.min "Permalink to this definition")
    :   Compute the minimum value along the specified axis

        other: matrix or numpy array (& other supported types) or scalar
        axis : int, optional

     `ml`{.descname}*= None*[](#systemml.matrix.ml "Permalink to this definition")
    :   

     `mod`{.descname}(*other*)[](#systemml.matrix.mod "Permalink to this definition")
    :   

     `ndim`{.descname}*= 2*[](#systemml.matrix.ndim "Permalink to this definition")
    :   

     `negative`{.descname}()[](#systemml.matrix.negative "Permalink to this definition")
    :   

     `ones_like`{.descname}()[](#systemml.matrix.ones_like "Permalink to this definition")
    :   

     `print_ast`{.descname}(*numSpaces=0*)[](#systemml.matrix.print_ast "Permalink to this definition")
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

     `rad2deg`{.descname}()[](#systemml.matrix.rad2deg "Permalink to this definition")
    :   Convert angles from radians to degrees.

     `reciprocal`{.descname}()[](#systemml.matrix.reciprocal "Permalink to this definition")
    :   

     `remainder`{.descname}(*other*)[](#systemml.matrix.remainder "Permalink to this definition")
    :   

     `round`{.descname}()[](#systemml.matrix.round "Permalink to this definition")
    :   

     `script`{.descname}*= None*[](#systemml.matrix.script "Permalink to this definition")
    :   

     `sd`{.descname}(*axis=None*)[](#systemml.matrix.sd "Permalink to this definition")
    :   Compute the standard deviation along the specified axis

        axis : int, optional

     `set_shape`{.descname}(*shape*)[](#systemml.matrix.set_shape "Permalink to this definition")
    :   

     `shape`{.descname}[](#systemml.matrix.shape "Permalink to this definition")
    :   

     `sign`{.descname}()[](#systemml.matrix.sign "Permalink to this definition")
    :   

     `sin`{.descname}()[](#systemml.matrix.sin "Permalink to this definition")
    :   

     `sqrt`{.descname}()[](#systemml.matrix.sqrt "Permalink to this definition")
    :   

     `square`{.descname}()[](#systemml.matrix.square "Permalink to this definition")
    :   

     `sum`{.descname}(*axis=None*)[](#systemml.matrix.sum "Permalink to this definition")
    :   Compute the sum along the specified axis

        axis : int, optional

     `systemmlVarID`{.descname}*= 0*[](#systemml.matrix.systemmlVarID "Permalink to this definition")
    :   

     `tan`{.descname}()[](#systemml.matrix.tan "Permalink to this definition")
    :   

     `toDataFrame`{.descname}()[](#systemml.matrix.toDataFrame "Permalink to this definition")
    :   This is a convenience function that calls the global eval method
        and then converts the matrix object into DataFrame.

     `toNumPyArray`{.descname}()[](#systemml.matrix.toNumPyArray "Permalink to this definition")
    :   This is a convenience function that calls the global eval method
        and then converts the matrix object into NumPy array.

     `toPandas`{.descname}()[](#systemml.matrix.toPandas "Permalink to this definition")
    :   This is a convenience function that calls the global eval method
        and then converts the matrix object into Pandas DataFrame.

     `trace`{.descname}()[](#systemml.matrix.trace "Permalink to this definition")
    :   Return the sum of the cells of the main diagonal square matrix

     `transpose`{.descname}()[](#systemml.matrix.transpose "Permalink to this definition")
    :   Transposes the matrix.

     `var`{.descname}(*axis=None*)[](#systemml.matrix.var "Permalink to this definition")
    :   Compute the variance along the specified axis

        axis : int, optional

     `visited`{.descname}*= []*[](#systemml.matrix.visited "Permalink to this definition")
    :   

     `zeros_like`{.descname}()[](#systemml.matrix.zeros_like "Permalink to this definition")
    :   

 `systemml.`{.descclassname}`eval`{.descname}(*outputs*, *outputDF=False*, *execute=True*)[](#systemml.eval "Permalink to this definition")
:   Executes the unevaluated DML script and computes the matrices
    specified by outputs.

    outputs: list of matrices or a matrix object outputDF: back the data
    of matrix as PySpark DataFrame

 `systemml.`{.descclassname}`solve`{.descname}(*A*, *b*)[](#systemml.solve "Permalink to this definition")
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

 *class*`systemml.`{.descclassname}`DMLOp`{.descname}(*inputs*, *dml=None*)[](#systemml.DMLOp "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Represents an intermediate node of Abstract syntax tree created to
    generate the PyDML script

     `MAX_DEPTH`{.descname}*= 0*[](#systemml.DMLOp.MAX_DEPTH "Permalink to this definition")
    :   

     `print_ast`{.descname}(*numSpaces*)[](#systemml.DMLOp.print_ast "Permalink to this definition")
    :   

 `systemml.`{.descclassname}`set_max_depth`{.descname}(*depth*)[](#systemml.set_max_depth "Permalink to this definition")
:   This method allows users to set the depth of lazy SystemML DAG.
    Setting this to 1 ensures that every operation is evaluated (which
    can turn off many optimization). Setting this to 0 ensures that
    SystemML will never implicitly evaluate the DAG, but will require
    user to call eval or toPandas/toNumPyArray/... explicitly.

    depth: Should be greater than equal to 0.

 `systemml.`{.descclassname}`debug_array_conversion`{.descname}(*throwError*)[](#systemml.debug_array_conversion "Permalink to this definition")
:   

 `systemml.`{.descclassname}`getNumCols`{.descname}(*numPyArr*)[](#systemml.getNumCols "Permalink to this definition")
:   

 `systemml.`{.descclassname}`convertToMatrixBlock`{.descname}(*sc*, *src*)[](#systemml.convertToMatrixBlock "Permalink to this definition")
:   

 `systemml.`{.descclassname}`convertToNumPyArr`{.descname}(*sc*, *mb*)[](#systemml.convertToNumPyArr "Permalink to this definition")
:   

 `systemml.`{.descclassname}`convertToPandasDF`{.descname}(*X*)[](#systemml.convertToPandasDF "Permalink to this definition")
:   

 `systemml.`{.descclassname}`convertToLabeledDF`{.descname}(*sqlCtx*, *X*, *y=None*)[](#systemml.convertToLabeledDF "Permalink to this definition")
:   



systemml.mllearn package(#systemml-mllearn-package "Permalink to this headline")
===================================================================================

Submodules(#submodules "Permalink to this headline")
-------------------------------------------------------

systemml.mllearn.estimators module(#module-systemml.mllearn.estimators "Permalink to this headline")
-------------------------------------------------------------------------------------------------------

 *class*`systemml.mllearn.estimators.`{.descclassname}`LinearRegression`{.descname}(*sqlCtx*, *fit\_intercept=True*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.estimators.LinearRegression "Permalink to this definition")
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

 *class*`systemml.mllearn.estimators.`{.descclassname}`LogisticRegression`{.descname}(*sqlCtx*, *penalty='l2'*, *fit\_intercept=True*, *max\_iter=100*, *max\_inner\_iter=0*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.estimators.LogisticRegression "Permalink to this definition")
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

 *class*`systemml.mllearn.estimators.`{.descclassname}`SVM`{.descname}(*sqlCtx*, *fit\_intercept=True*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *is\_multi\_class=False*, *transferUsingDF=False*)(#systemml.mllearn.estimators.SVM "Permalink to this definition")
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

 *class*`systemml.mllearn.estimators.`{.descclassname}`NaiveBayes`{.descname}(*sqlCtx*, *laplace=1.0*, *transferUsingDF=False*)(#systemml.mllearn.estimators.NaiveBayes "Permalink to this definition")
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

Module contents(#module-systemml.mllearn "Permalink to this headline")
-------------------------------------------------------------------------

### SystemML Algorithms(#systemml-algorithms "Permalink to this headline")

Classification Algorithms

LogisticRegression

Performs binomial and multinomial logistic regression

SVM

Performs both binary-class and multi-class SVM

NaiveBayes

Multinomial naive bayes classifier

Regression Algorithms

LinearRegression

Performs linear regression

 *class*`systemml.mllearn.`{.descclassname}`LinearRegression`{.descname}(*sqlCtx*, *fit\_intercept=True*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.LinearRegression "Permalink to this definition")
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

 *class*`systemml.mllearn.`{.descclassname}`LogisticRegression`{.descname}(*sqlCtx*, *penalty='l2'*, *fit\_intercept=True*, *max\_iter=100*, *max\_inner\_iter=0*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.LogisticRegression "Permalink to this definition")
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

 *class*`systemml.mllearn.`{.descclassname}`SVM`{.descname}(*sqlCtx*, *fit\_intercept=True*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *is\_multi\_class=False*, *transferUsingDF=False*)(#systemml.mllearn.SVM "Permalink to this definition")
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

 *class*`systemml.mllearn.`{.descclassname}`NaiveBayes`{.descname}(*sqlCtx*, *laplace=1.0*, *transferUsingDF=False*)(#systemml.mllearn.NaiveBayes "Permalink to this definition")
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



systemml.random package(#systemml-random-package "Permalink to this headline")
=================================================================================

Submodules(#submodules "Permalink to this headline")
-------------------------------------------------------

systemml.random.sampling module(#module-systemml.random.sampling "Permalink to this headline")
-------------------------------------------------------------------------------------------------

 `systemml.random.sampling.`{.descclassname}`normal`{.descname}(*loc=0.0*, *scale=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.sampling.normal "Permalink to this definition")
:   Draw random samples from a normal (Gaussian) distribution.

    loc: Mean (â€œcentreâ€) of the distribution. scale: Standard deviation
    (spread or â€œwidthâ€) of the distribution. size: Output shape (only
    tuple of length 2, i.e. (m, n), supported). sparsity: Sparsity
    (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.normal(loc=3, scale=2, size=(3,3))
        >>> m1.toNumPyArray()
        array([[ 3.48857226,  6.17261819,  2.51167259],
               [ 3.60506708, -1.90266305,  3.97601633],
               [ 3.62245706,  5.9430881 ,  2.53070413]])

 `systemml.random.sampling.`{.descclassname}`uniform`{.descname}(*low=0.0*, *high=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.sampling.uniform "Permalink to this definition")
:   Draw samples from a uniform distribution.

    low: Lower boundary of the output interval. high: Upper boundary of
    the output interval. size: Output shape (only tuple of length 2,
    i.e. (m, n), supported). sparsity: Sparsity (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.uniform(size=(3,3))
        >>> m1.toNumPyArray()
        array([[ 0.54511396,  0.11937437,  0.72975775],
               [ 0.14135946,  0.01944448,  0.52544478],
               [ 0.67582422,  0.87068849,  0.02766852]])

 `systemml.random.sampling.`{.descclassname}`poisson`{.descname}(*lam=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.sampling.poisson "Permalink to this definition")
:   Draw samples from a Poisson distribution.

    lam: Expectation of interval, should be \> 0. size: Output shape
    (only tuple of length 2, i.e. (m, n), supported). sparsity: Sparsity
    (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.poisson(lam=1, size=(3,3))
        >>> m1.toNumPyArray()
        array([[ 1.,  0.,  2.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  0.]])

Module contents(#module-systemml.random "Permalink to this headline")
------------------------------------------------------------------------

### Random Number Generation(#random-number-generation "Permalink to this headline")

Univariate distributions

normal

Normal / Gaussian distribution.

poisson

Poisson distribution.

uniform

Uniform distribution.

 `systemml.random.`{.descclassname}`normal`{.descname}(*loc=0.0*, *scale=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.normal "Permalink to this definition")
:   Draw random samples from a normal (Gaussian) distribution.

    loc: Mean (â€œcentreâ€) of the distribution. scale: Standard deviation
    (spread or â€œwidthâ€) of the distribution. size: Output shape (only
    tuple of length 2, i.e. (m, n), supported). sparsity: Sparsity
    (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.normal(loc=3, scale=2, size=(3,3))
        >>> m1.toNumPyArray()
        array([[ 3.48857226,  6.17261819,  2.51167259],
               [ 3.60506708, -1.90266305,  3.97601633],
               [ 3.62245706,  5.9430881 ,  2.53070413]])

 `systemml.random.`{.descclassname}`uniform`{.descname}(*low=0.0*, *high=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.uniform "Permalink to this definition")
:   Draw samples from a uniform distribution.

    low: Lower boundary of the output interval. high: Upper boundary of
    the output interval. size: Output shape (only tuple of length 2,
    i.e. (m, n), supported). sparsity: Sparsity (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.uniform(size=(3,3))
        >>> m1.toNumPyArray()
        array([[ 0.54511396,  0.11937437,  0.72975775],
               [ 0.14135946,  0.01944448,  0.52544478],
               [ 0.67582422,  0.87068849,  0.02766852]])

 `systemml.random.`{.descclassname}`poisson`{.descname}(*lam=1.0*, *size=(1*, *1)*, *sparsity=1.0*)(#systemml.random.poisson "Permalink to this definition")
:   Draw samples from a Poisson distribution.

    lam: Expectation of interval, should be \> 0. size: Output shape
    (only tuple of length 2, i.e. (m, n), supported). sparsity: Sparsity
    (between 0.0 and 1.0).

        >>> import systemml as sml
        >>> import numpy as np
        >>> sml.setSparkContext(sc)
        >>> from systemml import random
        >>> m1 = sml.random.poisson(lam=1, size=(3,3))
        >>> m1.toNumPyArray()
        array([[ 1.,  0.,  2.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  0.]])




