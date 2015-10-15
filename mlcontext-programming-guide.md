---
layout: global
title: MLContext Programming Guide
description: MLContext Programming Guide
---

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>


# Overview

The `MLContext` API offers a programmatic interface for interacting with SystemML from languages
such as Scala and Java. When interacting with `MLContext` from Spark, `DataFrame`s and `RDD`s can be passed
to SystemML. These data representations are converted to a
binary-block data format, allowing for SystemML's optimizations to be performed.


# Spark Shell Example

## Start Spark Shell with SystemML

To use SystemML with the Spark Shell, the SystemML jar can be referenced using the Spark Shell's `--jars` option. 
Instructions to build the SystemML jar can be found in the [SystemML GitHub README](http://www.github.com/SparkTC/systemml).

{% highlight bash %}
./bin/spark-shell --executor-memory 4G --driver-memory 4G --jars system-ml-5.2-SNAPSHOT.jar
{% endhighlight %}

Here is an example of Spark Shell with SystemML and YARN.

{% highlight bash %}
./bin/spark-shell --master yarn-client --num-executors 3 --driver-memory 5G --executor-memory 5G --executor-cores 4 --jars system-ml-5.2-SNAPSHOT.jar
{% endhighlight %}


## Create MLContext

An `MLContext` object can be created by passing its constructor a reference to the `SparkContext`.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala>import com.ibm.bi.dml.api.MLContext
import com.ibm.bi.dml.api.MLContext

scala> val ml = new MLContext(sc)
ml: com.ibm.bi.dml.api.MLContext = com.ibm.bi.dml.api.MLContext@33e38c6b
{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
import com.ibm.bi.dml.api.MLContext
val ml = new MLContext(sc)
{% endhighlight %}
</div>

</div>


## Create DataFrame

For demonstration purposes, we'll create a `DataFrame` consisting of 100,000 rows and 1,000 columns
of random `double`s.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> import org.apache.spark.sql._
import org.apache.spark.sql._

scala> import org.apache.spark.sql.types.{StructType,StructField,DoubleType}
import org.apache.spark.sql.types.{StructType, StructField, DoubleType}

scala> import scala.util.Random
import scala.util.Random

scala> val numRows = 100000
numRows: Int = 100000

scala> val numCols = 1000
numCols: Int = 1000

scala> val data = sc.parallelize(0 to numRows-1).map { _ => Row.fromSeq(Seq.fill(numCols)(Random.nextDouble)) }
data: org.apache.spark.rdd.RDD[org.apache.spark.sql.Row] = MapPartitionsRDD[1] at map at <console>:33

scala> val schema = StructType((0 to numCols-1).map { i => StructField("C" + i, DoubleType, true) } )
schema: org.apache.spark.sql.types.StructType = StructType(StructField(C0,DoubleType,true), StructField(C1,DoubleType,true), StructField(C2,DoubleType,true), StructField(C3,DoubleType,true), StructField(C4,DoubleType,true), StructField(C5,DoubleType,true), StructField(C6,DoubleType,true), StructField(C7,DoubleType,true), StructField(C8,DoubleType,true), StructField(C9,DoubleType,true), StructField(C10,DoubleType,true), StructField(C11,DoubleType,true), StructField(C12,DoubleType,true), StructField(C13,DoubleType,true), StructField(C14,DoubleType,true), StructField(C15,DoubleType,true), StructField(C16,DoubleType,true), StructField(C17,DoubleType,true), StructField(C18,DoubleType,true), StructField(C19,DoubleType,true), StructField(C20,DoubleType,true), StructField(C21,DoubleType,true), ...

scala> val df = sqlContext.createDataFrame(data, schema)
df: org.apache.spark.sql.DataFrame = [C0: double, C1: double, C2: double, C3: double, C4: double, C5: double, C6: double, C7: double, C8: double, C9: double, C10: double, C11: double, C12: double, C13: double, C14: double, C15: double, C16: double, C17: double, C18: double, C19: double, C20: double, C21: double, C22: double, C23: double, C24: double, C25: double, C26: double, C27: double, C28: double, C29: double, C30: double, C31: double, C32: double, C33: double, C34: double, C35: double, C36: double, C37: double, C38: double, C39: double, C40: double, C41: double, C42: double, C43: double, C44: double, C45: double, C46: double, C47: double, C48: double, C49: double, C50: double, C51: double, C52: double, C53: double, C54: double, C55: double, C56: double, C57: double, C58: double, C5...

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{StructType,StructField,DoubleType}
import scala.util.Random
val numRows = 100000
val numCols = 1000
val data = sc.parallelize(0 to numRows-1).map { _ => Row.fromSeq(Seq.fill(numCols)(Random.nextDouble)) }
val schema = StructType((0 to numCols-1).map { i => StructField("C" + i, DoubleType, true) } )
val df = sqlContext.createDataFrame(data, schema)
{% endhighlight %}
</div>

</div>


## Helper Methods

For convenience, we'll create some helper methods. The SystemML output data is encapsulated in
an `MLOutput` object. The `getScalar()` method extracts a scalar value from a `DataFrame` returned by
`MLOutput`. The `getScalarDouble()` method returns such a value as a `Double`, and the
`getScalarInt()` method returns such a value as an `Int`.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> import com.ibm.bi.dml.api.MLOutput
import com.ibm.bi.dml.api.MLOutput

scala> def getScalar(outputs: MLOutput, symbol: String): Any =
     | outputs.getDF(sqlContext, symbol).first()(1)
getScalar: (outputs: com.ibm.bi.dml.api.MLOutput, symbol: String)Any

scala> def getScalarDouble(outputs: MLOutput, symbol: String): Double =
     | getScalar(outputs, symbol).asInstanceOf[Double]
getScalarDouble: (outputs: com.ibm.bi.dml.api.MLOutput, symbol: String)Double

scala> def getScalarInt(outputs: MLOutput, symbol: String): Int =
     | getScalarDouble(outputs, symbol).toInt
getScalarInt: (outputs: com.ibm.bi.dml.api.MLOutput, symbol: String)Int

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
import com.ibm.bi.dml.api.MLOutput
def getScalar(outputs: MLOutput, symbol: String): Any =
outputs.getDF(sqlContext, symbol).first()(1)
def getScalarDouble(outputs: MLOutput, symbol: String): Double =
getScalar(outputs, symbol).asInstanceOf[Double]
def getScalarInt(outputs: MLOutput, symbol: String): Int =
getScalarDouble(outputs, symbol).toInt

{% endhighlight %}
</div>

</div>


## Convert DataFrame to Binary-Block Matrix

SystemML is optimized to operate on a binary-block format for matrix representation. For large
datasets, conversion from DataFrame to binary-block can require a significant quantity of time.
Explicit DataFrame to binary-block conversion allows algorithm performance to be measured separately
from data conversion time.

The SystemML binary-block matrix representation can be thought of as a two-dimensional array of blocks, where each block
consists of a number of rows and columns. In this example, we specify a matrix consisting
of blocks of size 1000x1000. The experimental `dataFrameToBinaryBlock()` method of `RDDConverterUtilsExt` is used
to convert the `DataFrame df` to a SystemML binary-block matrix, which is represented by the datatype
`JavaPairRDD[MatrixIndexes, MatrixBlock]`.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> import com.ibm.bi.dml.runtime.instructions.spark.utils.{RDDConverterUtilsExt => RDDConverterUtils}
import com.ibm.bi.dml.runtime.instructions.spark.utils.{RDDConverterUtilsExt=>RDDConverterUtils}

scala> import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics

scala> val numRowsPerBlock = 1000
numRowsPerBlock: Int = 1000

scala> val numColsPerBlock = 1000
numColsPerBlock: Int = 1000

scala> val mc = new MatrixCharacteristics(numRows, numCols, numRowsPerBlock, numColsPerBlock)
mc: com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics = [100000 x 1000, nnz=-1, blocks (1000 x 1000)]

scala> val sysMlMatrix = RDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc, false)
sysMlMatrix: org.apache.spark.api.java.JavaPairRDD[com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes,com.ibm.bi.dml.runtime.matrix.data.MatrixBlock] = org.apache.spark.api.java.JavaPairRDD@2bce3248

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
import com.ibm.bi.dml.runtime.instructions.spark.utils.{RDDConverterUtilsExt => RDDConverterUtils}
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
val numRowsPerBlock = 1000
val numColsPerBlock = 1000
val mc = new MatrixCharacteristics(numRows, numCols, numRowsPerBlock, numColsPerBlock)
val sysMlMatrix = RDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc, false)

{% endhighlight %}
</div>

</div>


## DML Script

For this example, we will utilize the following DML Script called `shape.dml` that reads in a matrix and outputs the number of rows and the
number of columns, each represented as a matrix.

{% highlight r %}
X = read($Xin)
m = matrix(nrow(X), rows=1, cols=1)
n = matrix(ncol(X), rows=1, cols=1)
write(m, $Mout)
write(n, $Nout)
{% endhighlight %}


## Execute Script

Let's execute our DML script, as shown in the example below. The call to `reset()` of `MLContext` is not necessary here, but this method should
be called if you need to reset inputs and outputs or if you would like to call `execute()` with a different script.

An example of registering the `DataFrame df` as an input to the `X` variable is shown but commented out. If a DataFrame is registered directly,
it will implicitly be converted to SystemML's binary-block format. However, since we've already explicitly converted the DataFrame to the
binary-block fixed variable `systemMlMatrix`, we will register this input to the `X` variable. We register the `m` and `n` variables
as outputs.

When SystemML is executed via `DMLScript` (such as in Standalone Mode), inputs are supplied as either command-line named arguments
or positional argument. These inputs are specified in DML scripts by prepending them with a `$`. Values are read from or written
to files using `read`/`write` (DML) and `load`/`save` (PyDML) statements. When utilizing the `MLContext` API,
inputs and outputs can be other data representations, such as `DataFrame`s. The input and output data are bound to DML variables.
The named arguments in the `shape.dml` script do not have default values set for them, so we create a `Map` to map the required named
arguments to blank `String`s so that the script can pass validation.

The `shape.dml` script is executed by the call to `execute()`, where we supply the `Map` of required named arguments. The
execution results are returned as the `MLOutput` fixed variable `outputs`. The number of rows is obtained by calling the `getStaticInt()`
helper method with the `outputs` object and `"m"`. The number of columns is retrieved by calling `getStaticInt()` with 
`outputs` and `"n"`.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> ml.reset()

scala> //ml.registerInput("X", df) // implicit conversion of DataFrame to binary-block

scala> ml.registerInput("X", sysMlMatrix, numRows, numCols)

scala> ml.registerOutput("m")

scala> ml.registerOutput("n")

scala> val nargs = Map("Xin" -> " ", "Mout" -> " ", "Nout" -> " ")
nargs: scala.collection.immutable.Map[String,String] = Map(Xin -> " ", Mout -> " ", Nout -> " ")

scala> val outputs = ml.execute("shape.dml", nargs)
15/10/12 16:29:15 WARN : Your hostname, derons-mbp.usca.ibm.com resolves to a loopback/non-reachable address: 127.0.0.1, but we couldn't find any external IP address!
15/10/12 16:29:15 WARN OptimizerUtils: Auto-disable multi-threaded text read for 'text' and 'csv' due to thread contention on JRE < 1.8 (java.version=1.7.0_80).
outputs: com.ibm.bi.dml.api.MLOutput = com.ibm.bi.dml.api.MLOutput@4d424743

scala> val m = getScalarInt(outputs, "m")
m: Int = 100000

scala> val n = getScalarInt(outputs, "n")
n: Int = 1000

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
ml.reset()
//ml.registerInput("X", df) // implicit conversion of DataFrame to binary-block
ml.registerInput("X", sysMlMatrix, numRows, numCols)
ml.registerOutput("m")
ml.registerOutput("n")
val nargs = Map("Xin" -> " ", "Mout" -> " ", "Nout" -> " ")
val outputs = ml.execute("shape.dml", nargs)
val m = getScalarInt(outputs, "m")
val n = getScalarInt(outputs, "n")

{% endhighlight %}
</div>

</div>


## DML Script as String

The `MLContext` API allows a DML script to be specified
as a `String`. Here, we specify a DML script as a fixed `String` variable called `minMaxMeanScript`.
This DML will find the minimum, maximum, and mean value of a matrix.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> val minMaxMeanScript: String = 
     | """
     | Xin = read(" ")
     | minOut = matrix(min(Xin), rows=1, cols=1)
     | maxOut = matrix(max(Xin), rows=1, cols=1)
     | meanOut = matrix(mean(Xin), rows=1, cols=1)
     | write(minOut, " ")
     | write(maxOut, " ")
     | write(meanOut, " ")
     | """
minMaxMeanScript: String = 
"
Xin = read(" ")
minOut = matrix(min(Xin), rows=1, cols=1)
maxOut = matrix(max(Xin), rows=1, cols=1)
meanOut = matrix(mean(Xin), rows=1, cols=1)
write(minOut, " ")
write(maxOut, " ")
write(meanOut, " ")
"

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
val minMaxMeanScript: String = 
"""
Xin = read(" ")
minOut = matrix(min(Xin), rows=1, cols=1)
maxOut = matrix(max(Xin), rows=1, cols=1)
meanOut = matrix(mean(Xin), rows=1, cols=1)
write(minOut, " ")
write(maxOut, " ")
write(meanOut, " ")
"""

{% endhighlight %}
</div>

</div>

## Scala Wrapper for DML

We can create a Scala wrapper for our invocation of the `minMaxMeanScript` DML `String`. The `minMaxMean()` method
takes a `JavaPairRDD[MatrixIndexes, MatrixBlock]` parameter, which is a SystemML binary-block matrix representation.
It also takes a `rows` parameter indicating the number of rows in the matrix, a `cols` parameter indicating the number
of columns in the matrix, and an `MLContext` parameter. The `minMaxMean()` method
returns a tuple consisting of the minimum value in the matrix, the maximum value in the matrix, and the computed
mean value of the matrix.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes

scala> import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock

scala> import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.api.java.JavaPairRDD

scala> def minMaxMean(mat: JavaPairRDD[MatrixIndexes, MatrixBlock], rows: Int, cols: Int, ml: MLContext): (Double, Double, Double) = {
     | ml.reset()
     | ml.registerInput("Xin", mat, rows, cols)
     | ml.registerOutput("minOut")
     | ml.registerOutput("maxOut")
     | ml.registerOutput("meanOut")
     | val outputs = ml.executeScript(minMaxMeanScript)
     | val minOut = getScalarDouble(outputs, "minOut")
     | val maxOut = getScalarDouble(outputs, "maxOut")
     | val meanOut = getScalarDouble(outputs, "meanOut")
     | (minOut, maxOut, meanOut)
     | }
minMaxMean: (mat: org.apache.spark.api.java.JavaPairRDD[com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes,com.ibm.bi.dml.runtime.matrix.data.MatrixBlock], rows: Int, cols: Int, ml: com.ibm.bi.dml.api.MLContext)(Double, Double, Double)

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock
import org.apache.spark.api.java.JavaPairRDD
def minMaxMean(mat: JavaPairRDD[MatrixIndexes, MatrixBlock], rows: Int, cols: Int, ml: MLContext): (Double, Double, Double) = {
ml.reset()
ml.registerInput("Xin", mat, rows, cols)
ml.registerOutput("minOut")
ml.registerOutput("maxOut")
ml.registerOutput("meanOut")
val outputs = ml.executeScript(minMaxMeanScript)
val minOut = getScalarDouble(outputs, "minOut")
val maxOut = getScalarDouble(outputs, "maxOut")
val meanOut = getScalarDouble(outputs, "meanOut")
(minOut, maxOut, meanOut)
}

{% endhighlight %}
</div>

</div>


## Invoking DML via Scala Wrapper

Here, we invoke `minMaxMeanScript` using our `minMaxMean()` Scala wrapper method. It returns a tuple
consisting of the minimum value in the matrix, the maximum value in the matrix, and the mean value of the matrix.

<div class="codetabs">

<div data-lang="Spark Shell" markdown="1">
{% highlight scala %}
scala> val (min, max, mean) = minMaxMean(sysMlMatrix, numRows, numCols, ml)
15/10/13 14:33:11 WARN OptimizerUtils: Auto-disable multi-threaded text read for 'text' and 'csv' due to thread contention on JRE < 1.8 (java.version=1.7.0_80).
min: Double = 5.378949397005783E-9                                              
max: Double = 0.9999999934660398
mean: Double = 0.499988222338507

{% endhighlight %}
</div>

<div data-lang="Statements" markdown="1">
{% highlight scala %}
val (min, max, mean) = minMaxMean(sysMlMatrix, numRows, numCols, ml)

{% endhighlight %}
</div>

</div>


* * *

# Java Example

Next, let's consider a Java example. The `MLContextExample` class creates an `MLContext` object from a `JavaSparkContext`.
Next, it reads in a matrix CSV file as a `JavaRDD<String>` object. It registers this as input `X`. It registers
two outputs, `m` and `n`. A `HashMap` maps the expected command-line arguments of the `shape.dml` script to spaces so that
it passes validation. The `shape.dml` script is executed, and the number of rows and columns in the matrix are output
to standard output.


{% highlight java %}
package com.ibm.bi.dml;

import java.util.HashMap;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.api.MLOutput;

public class MLContextExample {

	public static void main(String[] args) throws Exception {

		SparkConf conf = new SparkConf().setAppName("MLContextExample").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(sc);
		MLContext ml = new MLContext(sc);

		JavaRDD<String> csv = sc.textFile("A.csv");
		ml.registerInput("X", csv, "csv");
		ml.registerOutput("m");
		ml.registerOutput("n");
		HashMap<String, String> cmdLineArgs = new HashMap<String, String>();
		cmdLineArgs.put("X", " ");
		cmdLineArgs.put("m", " ");
		cmdLineArgs.put("n", " ");
		MLOutput output = ml.execute("shape.dml", cmdLineArgs);
		DataFrame mDf = output.getDF(sqlContext, "m");
		DataFrame nDf = output.getDF(sqlContext, "n");
		System.out.println("rows:" + mDf.first().getDouble(1));
		System.out.println("cols:" + nDf.first().getDouble(1));
	}

}


{% endhighlight %}


* * *

# Zeppelin Notebook Example - Linear Regression Algorithm

Next, we'll consider an example of a SystemML linear regression algorithm run from Spark through an Apache Zeppelin notebook.
Instructions to clone and build Zeppelin can be found at the [GitHub Apache Zeppelin](https://github.com/apache/incubator-zeppelin)
site. This example also will look at the Spark ML linear regression algorithm.

This Zeppelin notebook example can be downloaded [here](files/mlcontext-programming-guide/zeppelin-notebook-linear-regression/2AZ2AQ12B.tar.gz).
Once downloaded and unzipped, place the folder in the Zeppelin `notebook` directory.

A `conf/zeppelin-env.sh` file is created based on `conf/zeppelin-env.sh.template`. For
this demonstration, it features `SPARK_HOME`, `SPARK_SUBMIT_OPTIONS`, and `ZEPPELIN_SPARK_USEHIVECONTEXT`
environment variables:

	export SPARK_HOME=/Users/example/spark-1.5.1-bin-hadoop2.6
	export SPARK_SUBMIT_OPTIONS="--jars $/Users/example/systemml/system-ml/target/system-ml-5.2-SNAPSHOT.jar"
	export ZEPPELIN_SPARK_USEHIVECONTEXT=false

Start Zeppelin using the `zeppelin.sh` script:

	bin/zeppelin.sh

After opening Zeppelin in a brower, we see the "SystemML - Linear Regression" note in the list of available
Zeppelin notes.

![Zeppelin Notebook](img/mlcontext-programming-guide/zeppelin-notebook.png "Zeppelin Notebook")

If we go to the "SystemML - Linear Regression" note, we see that the note consists of several cells of code.

![Zeppelin 'SystemML - Linear Regression' Note](img/mlcontext-programming-guide/zeppelin-notebook-systemml-linear-regression.png "Zeppelin 'SystemML - Linear Regression' Note")

Let's briefly consider these cells.

## Trigger Spark Startup

This cell triggers Spark to initialize by calling the `SparkContext` `sc` object. Information regarding these startup operations can be viewed in the 
console window in which `zeppelin.sh` is running.

**Cell:**
{% highlight scala %}
// Trigger Spark Startup
sc
{% endhighlight %}

**Output:**
{% highlight scala %}
res8: org.apache.spark.SparkContext = org.apache.spark.SparkContext@6ce70bf3
{% endhighlight %}


## Generate Linear Regression Test Data

The Spark `LinearDataGenerator` is used to generate test data for the Spark ML and SystemML linear regression algorithms.

**Cell:**
{% highlight scala %}
// Generate data
import org.apache.spark.mllib.util.LinearDataGenerator

val numRows = 10000
val numCols = 1000
val rawData = LinearDataGenerator.generateLinearRDD(sc, numRows, numCols, 1).toDF()

// Repartition into a more parallelism-friendly number of partitions
val data = rawData.repartition(64).cache()
{% endhighlight %}

**Output:**
{% highlight scala %}
import org.apache.spark.mllib.util.LinearDataGenerator
numRows: Int = 10000
numCols: Int = 1000
rawData: org.apache.spark.sql.DataFrame = [label: double, features: vector]
data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
{% endhighlight %}


## Train using Spark ML Linear Regression Algorithm for Comparison

For purpose of comparison, we can train a model using the Spark ML linear regression
algorithm.

**Cell:**
{% highlight scala %}
// Spark ML
import org.apache.spark.ml.regression.LinearRegression

// Model Settings
val maxIters = 100
val reg = 0
val elasticNetParam = 0  // L2 reg

// Fit the model
val lr = new LinearRegression()
  .setMaxIter(maxIters)
  .setRegParam(reg)
  .setElasticNetParam(elasticNetParam)
val start = System.currentTimeMillis()
val model = lr.fit(data)
val trainingTime = (System.currentTimeMillis() - start).toDouble / 1000.0

// Summarize the model over the training set and gather some metrics
val trainingSummary = model.summary
val r2 = trainingSummary.r2
val iters = trainingSummary.totalIterations
val trainingTimePerIter = trainingTime / iters
{% endhighlight %}

**Output:**
{% highlight scala %}
import org.apache.spark.ml.regression.LinearRegression
maxIters: Int = 100
reg: Int = 0
elasticNetParam: Int = 0
lr: org.apache.spark.ml.regression.LinearRegression = linReg_a7f51d676562
start: Long = 1444672044647
model: org.apache.spark.ml.regression.LinearRegressionModel = linReg_a7f51d676562
trainingTime: Double = 12.985
trainingSummary: org.apache.spark.ml.regression.LinearRegressionTrainingSummary = org.apache.spark.ml.regression.LinearRegressionTrainingSummary@227ba28b
r2: Double = 0.9677118209276552
iters: Int = 17
trainingTimePerIter: Double = 0.7638235294117647
{% endhighlight %}


## Spark ML Linear Regression Summary Statistics

Summary statistics for the Spark ML linear regression algorithm are displayed by this cell.

**Cell:**
{% highlight scala %}
// Print statistics
println(s"R2: ${r2}")
println(s"Iterations: ${iters}")
println(s"Training time per iter: ${trainingTimePerIter} seconds")
{% endhighlight %}

**Output:**
{% highlight scala %}
R2: 0.9677118209276552
Iterations: 17
Training time per iter: 0.7638235294117647 seconds
{% endhighlight %}


## SystemML Linear Regression Algorithm

The `linearReg` fixed `String` variable is set to
a linear regression algorithm written in DML, SystemML's Declarative Machine Learning language.



**Cell:**
{% highlight scala %}
// SystemML kernels
val linearReg =
"""
#
# THIS SCRIPT SOLVES LINEAR REGRESSION USING THE CONJUGATE GRADIENT ALGORITHM
#
# INPUT PARAMETERS:
# --------------------------------------------------------------------------------------------
# NAME  TYPE   DEFAULT  MEANING
# --------------------------------------------------------------------------------------------
# X     String  ---     Matrix X of feature vectors
# Y     String  ---     1-column Matrix Y of response values
# icpt  Int      0      Intercept presence, shifting and rescaling the columns of X:
#                       0 = no intercept, no shifting, no rescaling;
#                       1 = add intercept, but neither shift nor rescale X;
#                       2 = add intercept, shift & rescale X columns to mean = 0, variance = 1
# reg   Double 0.000001 Regularization constant (lambda) for L2-regularization; set to nonzero
#                       for highly dependend/sparse/numerous features
# tol   Double 0.000001 Tolerance (epsilon); conjugate graduent procedure terminates early if
#                       L2 norm of the beta-residual is less than tolerance * its initial norm
# maxi  Int      0      Maximum number of conjugate gradient iterations, 0 = no maximum
# --------------------------------------------------------------------------------------------
#
# OUTPUT:
# B Estimated regression parameters (the betas) to store
#
# Note: Matrix of regression parameters (the betas) and its size depend on icpt input value:
#         OUTPUT SIZE:   OUTPUT CONTENTS:                HOW TO PREDICT Y FROM X AND B:
# icpt=0: ncol(X)   x 1  Betas for X only                Y ~ X %*% B[1:ncol(X), 1], or just X %*% B
# icpt=1: ncol(X)+1 x 1  Betas for X and intercept       Y ~ X %*% B[1:ncol(X), 1] + B[ncol(X)+1, 1]
# icpt=2: ncol(X)+1 x 2  Col.1: betas for X & intercept  Y ~ X %*% B[1:ncol(X), 1] + B[ncol(X)+1, 1]
#                        Col.2: betas for shifted/rescaled X and intercept
#

fileX = "";
fileY = "";
fileB = "";

intercept_status = ifdef ($icpt, 0);     # $icpt=0;
tolerance = ifdef ($tol, 0.000001);      # $tol=0.000001;
max_iteration = ifdef ($maxi, 0);        # $maxi=0;
regularization = ifdef ($reg, 0.000001); # $reg=0.000001;

X = read (fileX);
y = read (fileY);

n = nrow (X);
m = ncol (X);
ones_n = matrix (1, rows = n, cols = 1);
zero_cell = matrix (0, rows = 1, cols = 1);

# Introduce the intercept, shift and rescale the columns of X if needed

m_ext = m;
if (intercept_status == 1 | intercept_status == 2)  # add the intercept column
{
    X = append (X, ones_n);
    m_ext = ncol (X);
}

scale_lambda = matrix (1, rows = m_ext, cols = 1);
if (intercept_status == 1 | intercept_status == 2)
{
    scale_lambda [m_ext, 1] = 0;
}

if (intercept_status == 2)  # scale-&-shift X columns to mean 0, variance 1
{                           # Important assumption: X [, m_ext] = ones_n
    avg_X_cols = t(colSums(X)) / n;
    var_X_cols = (t(colSums (X ^ 2)) - n * (avg_X_cols ^ 2)) / (n - 1);
    is_unsafe = ppred (var_X_cols, 0.0, "<=");
    scale_X = 1.0 / sqrt (var_X_cols * (1 - is_unsafe) + is_unsafe);
    scale_X [m_ext, 1] = 1;
    shift_X = - avg_X_cols * scale_X;
    shift_X [m_ext, 1] = 0;
} else {
    scale_X = matrix (1, rows = m_ext, cols = 1);
    shift_X = matrix (0, rows = m_ext, cols = 1);
}

# Henceforth, if intercept_status == 2, we use "X %*% (SHIFT/SCALE TRANSFORM)"
# instead of "X".  However, in order to preserve the sparsity of X,
# we apply the transform associatively to some other part of the expression
# in which it occurs.  To avoid materializing a large matrix, we rewrite it:
#
# ssX_A  = (SHIFT/SCALE TRANSFORM) %*% A    --- is rewritten as:
# ssX_A  = diag (scale_X) %*% A;
# ssX_A [m_ext, ] = ssX_A [m_ext, ] + t(shift_X) %*% A;
#
# tssX_A = t(SHIFT/SCALE TRANSFORM) %*% A   --- is rewritten as:
# tssX_A = diag (scale_X) %*% A + shift_X %*% A [m_ext, ];

lambda = scale_lambda * regularization;
beta_unscaled = matrix (0, rows = m_ext, cols = 1);

if (max_iteration == 0) {
    max_iteration = m_ext;
}
i = 0;

# BEGIN THE CONJUGATE GRADIENT ALGORITHM
r = - t(X) %*% y;

if (intercept_status == 2) {
    r = scale_X * r + shift_X %*% r [m_ext, ];
}

p = - r;
norm_r2 = sum (r ^ 2);
norm_r2_initial = norm_r2;
norm_r2_target = norm_r2_initial * tolerance ^ 2;

while (i < max_iteration & norm_r2 > norm_r2_target)
{
    if (intercept_status == 2) {
        ssX_p = scale_X * p;
        ssX_p [m_ext, ] = ssX_p [m_ext, ] + t(shift_X) %*% p;
    } else {
        ssX_p = p;
    }

    q = t(X) %*% (X %*% ssX_p);

    if (intercept_status == 2) {
        q = scale_X * q + shift_X %*% q [m_ext, ];
    }

    q = q + lambda * p;
    a = norm_r2 / sum (p * q);
    beta_unscaled = beta_unscaled + a * p;
    r = r + a * q;
    old_norm_r2 = norm_r2;
    norm_r2 = sum (r ^ 2);
    p = -r + (norm_r2 / old_norm_r2) * p;
    i = i + 1;
}
# END THE CONJUGATE GRADIENT ALGORITHM

if (intercept_status == 2) {
    beta = scale_X * beta_unscaled;
    beta [m_ext, ] = beta [m_ext, ] + t(shift_X) %*% beta_unscaled;
} else {
    beta = beta_unscaled;
}

# Output statistics
avg_tot = sum (y) / n;
ss_tot = sum (y ^ 2);
ss_avg_tot = ss_tot - n * avg_tot ^ 2;
var_tot = ss_avg_tot / (n - 1);
y_residual = y - X %*% beta;
avg_res = sum (y_residual) / n;
ss_res = sum (y_residual ^ 2);
ss_avg_res = ss_res - n * avg_res ^ 2;

R2_temp = 1 - ss_res / ss_avg_tot
R2 = matrix(R2_temp, rows=1, cols=1)
write(R2, "")

totalIters = matrix(i, rows=1, cols=1)
write(totalIters, "")

# Prepare the output matrix
if (intercept_status == 2) {
    beta_out = append (beta, beta_unscaled);
} else {
    beta_out = beta;
}

write (beta_out, fileB);
"""
{% endhighlight %}

**Output:**

None


## Helper Methods

This cell contains helper methods to return `Double` and `Int` values from output generated by the `MLContext` API.

**Cell:**
{% highlight scala %}
// Helper functions
import com.ibm.bi.dml.api.MLOutput

def getScalar(outputs: MLOutput, symbol: String): Any =
    outputs.getDF(sqlContext, symbol).first()(1)
    
def getScalarDouble(outputs: MLOutput, symbol: String): Double = 
    getScalar(outputs, symbol).asInstanceOf[Double]
    
def getScalarInt(outputs: MLOutput, symbol: String): Int =
    getScalarDouble(outputs, symbol).toInt
{% endhighlight %}

**Output:**
{% highlight scala %}
import com.ibm.bi.dml.api.MLOutput
getScalar: (outputs: com.ibm.bi.dml.api.MLOutput, symbol: String)Any
getScalarDouble: (outputs: com.ibm.bi.dml.api.MLOutput, symbol: String)Double
getScalarInt: (outputs: com.ibm.bi.dml.api.MLOutput, symbol: String)Int
{% endhighlight %}


## Convert DataFrame to Binary-Block Format

SystemML uses a binary-block format for matrix data representation. This cell
explicitly converts the `DataFrame` `data` object to a binary-block `features` matrix
and single-column `label` matrix, both represented by the
`JavaPairRDD[MatrixIndexes, MatrixBlock]` datatype. 


**Cell:**
{% highlight scala %}
// Imports
import com.ibm.bi.dml.api.MLContext
import com.ibm.bi.dml.runtime.instructions.spark.utils.{RDDConverterUtilsExt => RDDConverterUtils}
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;

// Create SystemML context
val ml = new MLContext(sc)

// Convert data to proper format
val mcX = new MatrixCharacteristics(numRows, numCols, 1000, 1000)
val mcY = new MatrixCharacteristics(numRows, 1, 1000, 1000)
val X = RDDConverterUtils.vectorDataFrameToBinaryBlock(sc, data, mcX, false, "features")
val y = RDDConverterUtils.dataFrameToBinaryBlock(sc, data.select("label"), mcY, false)
// val y = data.select("label")

// Cache
val X2 = X.cache()
val y2 = y.cache()
val cnt1 = X2.count()
val cnt2 = y2.count()
{% endhighlight %}

**Output:**
{% highlight scala %}
import com.ibm.bi.dml.api.MLContext
import com.ibm.bi.dml.runtime.instructions.spark.utils.{RDDConverterUtilsExt=>RDDConverterUtils}
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics
ml: com.ibm.bi.dml.api.MLContext = com.ibm.bi.dml.api.MLContext@38d59245
mcX: com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics = [10000 x 1000, nnz=-1, blocks (1000 x 1000)]
mcY: com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics = [10000 x 1, nnz=-1, blocks (1000 x 1000)]
X: org.apache.spark.api.java.JavaPairRDD[com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes,com.ibm.bi.dml.runtime.matrix.data.MatrixBlock] = org.apache.spark.api.java.JavaPairRDD@b5a86e3
y: org.apache.spark.api.java.JavaPairRDD[com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes,com.ibm.bi.dml.runtime.matrix.data.MatrixBlock] = org.apache.spark.api.java.JavaPairRDD@56377665
X2: org.apache.spark.api.java.JavaPairRDD[com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes,com.ibm.bi.dml.runtime.matrix.data.MatrixBlock] = org.apache.spark.api.java.JavaPairRDD@650f29d2
y2: org.apache.spark.api.java.JavaPairRDD[com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes,com.ibm.bi.dml.runtime.matrix.data.MatrixBlock] = org.apache.spark.api.java.JavaPairRDD@334857a8
cnt1: Long = 10
cnt2: Long = 10
{% endhighlight %}


## Train using SystemML Linear Regression Algorithm

Now, we can train our model using the SystemML linear regression algorithm. We register the features matrix `X` and the label matrix `y` as inputs. We register the `beta_out` matrix,
`R2`, and `totalIters` as outputs.

**Cell:**
{% highlight scala %}
// Register inputs & outputs
ml.reset()  
ml.registerInput("X", X, numRows, numCols)
ml.registerInput("y", y, numRows, 1)
// ml.registerInput("y", y)
ml.registerOutput("beta_out")
ml.registerOutput("R2")
ml.registerOutput("totalIters")

// Run the script
val start = System.currentTimeMillis()
val outputs = ml.executeScript(linearReg)
val trainingTime = (System.currentTimeMillis() - start).toDouble / 1000.0

// Get outputs
val B = outputs.getDF(sqlContext, "beta_out").sort("ID").drop("ID")
val r2 = getScalarDouble(outputs, "R2")
val iters = getScalarInt(outputs, "totalIters")
val trainingTimePerIter = trainingTime / iters
{% endhighlight %}

**Output:**
{% highlight scala %}
start: Long = 1444672090620
outputs: com.ibm.bi.dml.api.MLOutput = com.ibm.bi.dml.api.MLOutput@5d2c22d0
trainingTime: Double = 1.176
B: org.apache.spark.sql.DataFrame = [C1: double]
r2: Double = 0.9677079547216473
iters: Int = 12
trainingTimePerIter: Double = 0.09799999999999999
{% endhighlight %}


## SystemML Linear Regression Summary Statistics

SystemML linear regression summary statistics are displayed by this cell.

**Cell:**
{% highlight scala %}
// Print statistics
println(s"R2: ${r2}")
println(s"Iterations: ${iters}")
println(s"Training time per iter: ${trainingTimePerIter} seconds")
B.describe().show()
{% endhighlight %}

**Output:**
{% highlight scala %}
R2: 0.9677079547216473
Iterations: 12
Training time per iter: 0.2334166666666667 seconds
+-------+-------------------+
|summary|                 C1|
+-------+-------------------+
|  count|               1000|
|   mean| 0.0184500840658385|
| stddev| 0.2764750319432085|
|    min|-0.5426068958986378|
|    max| 0.5225309861616542|
+-------+-------------------+
{% endhighlight %}



