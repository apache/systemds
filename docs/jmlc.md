---
layout: global
title: Java Machine Learning Connector (JMLC)
description: Java Machine Learning Connector (JMLC)
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


# Overview

The `Java Machine Learning Connector (JMLC)` API is a programmatic interface for interacting with SystemML
in an embedded fashion. To use JMLC, the small footprint "in-memory" SystemML jar file needs to be included on the
classpath of the Java application, since JMLC invokes SystemML in an existing Java Virtual Machine. Because
of this, JMLC allows access to SystemML's optimizations and fast linear algebra, but the bulk performance
gain from running SystemML on a large Spark or Hadoop cluster is not available. However, this embeddable nature
allows SystemML to be part of a production pipeline for tasks such as scoring.

The primary purpose of JMLC is as a scoring API, where your scoring function is expressed using
SystemML's DML (Declarative Machine Learning) language. Scoring occurs on a single machine in a single
JVM on a relatively small amount of input data which produces a relatively small amount of output data.
For consistency, it is important to be able to express a scoring function in the same DML language used for
training a model, since different implementations of linear algebra (for instance MATLAB and R) can deliver
slightly different results.

In addition to scoring, embedded SystemML can be used for tasks such as unsupervised learning (for
example, clustering) in the context of a larger application running on a single machine.

Performance penalties include startup costs, so JMLC has facilities to perform some startup tasks once,
such as script precompilation. Due to startup costs, it tends to be best practice to do batch scoring, such
as scoring 1000 records at a time. For large amounts of data, it is recommended to run DML in one
of SystemML's distributed modes, such as Spark batch mode or Hadoop batch mode, to take advantage of SystemML's
distributed computing capabilities. JMLC offers embeddability at the cost of performance, so its use is
dependent on the nature of the business use case being addressed.


---

# Examples

JMLC is patterned loosely after JDBC. To interact with SystemML via JMLC, we can begin by creating a `Connection`
object. We can then prepare (precompile) a DML script by calling the `Connection`'s `prepareScript` method,
which returns a `PreparedScript` object. We can then call the `executeScript` method on the `PreparedScript`
object to invoke this script.

Here, we see a "hello world" example, which invokes SystemML via JMLC and prints "hello world" to the console.

{% highlight java %}
Connection conn = new Connection();
String dml = "print('hello world');";
PreparedScript script = conn.prepareScript(dml, new String[0], new String[0], false);
script.executeScript();
{% endhighlight %}

---

Next, let's consider a more practical example. Consider the following DML script. It takes input data matrix `X`
and input model matrix `W`. Scores are computed, and it returns a (n x 1) matrix `predicted_y` consisting of the
column indexes of the maximum values of each row in the `scores` matrix. Note that since values are being read
in and out programmatically, we can ignore the parameters in the `read` and `write` statements.

#### DML
{% highlight r %}
X = read("./tmp/X", rows=-1, cols=-1);
W = read("./tmp/W", rows=-1, cols=-1);

numRows = nrow(X);
numCols = ncol(X);
b = W[numCols+1,]
scores = X %*% W[1:numCols,] + b;
predicted_y = rowIndexMax(scores);

write(predicted_y, "./tmp", format="text");
{% endhighlight %}


In the Java below, we initialize SystemML by obtaining a `Connection` object. Next, we read in the above DML script
(`"scoring-example.dml"`) as a `String`. We precompile this script by calling the `prepareScript` method on the
`Connection` object with the names of the inputs (`"W"` and `"X"`) and outputs (`"predicted_y"`) to register.

Following this, we read in the model (`"sentiment_model.mtx"`) and convert the model to a 47x46 matrix, where the
last row of the matrix is for the `b` values. We set this matrix as the `"W"` input. Next, we create a random 46x46 matrix
of doubles for test data with a sparsity of 0.7 and set this matrix as the `"X"` input. We then execute the script and
read the `"predicted_y"` result matrix.


#### Java

{% highlight java %}
 package org.apache.sysml.example;
 
 import java.util.Random;
 
 import org.apache.sysml.api.jmlc.Connection;
 import org.apache.sysml.api.jmlc.PreparedScript;
 import org.apache.sysml.api.jmlc.ResultVariables;
 
 public class JMLCExample {
 
    public static void main(String[] args) throws Exception {
 
       // obtain connection to SystemML
       Connection conn = new Connection();
 
       // read in and precompile DML script, registering inputs and outputs
       String dml = conn.readScript("scoring-example.dml");
       PreparedScript script = conn.prepareScript(dml, new String[] { "W", "X" }, new String[] { "predicted_y" }, false);
 
      // read in model and set model
       String model = conn.readScript("sentiment_model.mtx");
       double[][] w = conn.convertToDoubleMatrix(model, 47, 46);
       script.setMatrix("W", w);
 
       // read in data and set data
       double[][] x = generateRandomMatrix(46, 46, -1, 1, 0.7, System.nanoTime());
       script.setMatrix("X", x);
 
       // execute script and get output
       ResultVariables results = script.executeScript();
       double[][] y = results.getMatrix("predicted_y");
 
       // close connection
       conn.close();
    }
 
    public static double[][] generateRandomMatrix(int rows, int cols, double min, double max, double sparsity, long seed) {
       double[][] matrix = new double[rows][cols];
       Random random = (seed == -1) ? new Random(System.currentTimeMillis()) : new Random(seed);
       for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
             if (random.nextDouble() > sparsity) {
                continue;
             }
             matrix[i][j] = (random.nextDouble() * (max - min) + min);
          }
       }
       return matrix;
    }
 }
{% endhighlight %}


---

For additional information regarding programmatic access to SystemML, please see the
[Spark MLContext Programming Guide](spark-mlcontext-programming-guide.html).
