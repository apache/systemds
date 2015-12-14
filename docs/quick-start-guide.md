---
layout: global
title: SystemML Quick Start Guide
author: Christian Kadner
description: SystemML Quick Start Guide
displayTitle: SystemML Quick Start Guide
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

This tutorial provides a quick introduction to using SystemML by
running existing SystemML algorithms in standalone mode. More information
about running SystemML in distributed execution mode (Hadoop, Spark) will 
be added soon.
For more in-depth information, please refer to the
[Algorithms Reference](algorithms-reference.html) and the
[DML Language Reference](dml-language-reference.html).


# What is SystemML

SystemML enables large-scale machine learning (ML) via a high-level declarative
language with R-like syntax called [DML](dml-language-reference.html). 
This language allows data scientists to 
express their ML algorithms with full flexibility but without the need to fine-tune 
distributed runtime execution plans and system configurations. 
These ML programs are dynamically compiled and optimized based on data 
and cluster characteristics using rule- and cost-based optimization techniques. 
The compiler automatically generates hybrid runtime execution plans ranging 
from in-memory, single node execution to distributed computation for Hadoop M/R 
or Spark Batch execution.
SystemML features a suite of algorithms for Descriptive Statistics, Classification, 
Clustering, Regression, Matrix Factorization, and Survival Analysis. Detailed descriptions of these 
algorithms can be found in the [Algorithms Reference](algorithms-reference.html).

<br/>

# Download SystemML

The pre-incubator binary release of SystemML 0.8.0 is available for download at
[https://github.com/SparkTC/systemml/releases](https://github.com/SparkTC/systemml/releases).
Apache incubator binary releases of SystemML will be available from the [Apache SystemML (incubating)](http://systemml.apache.org/) website.

The SystemML project is available on GitHub at [https://github.com/apache/incubator-systemml](https://github.com/apache/incubator-systemml).
SystemML can be downloaded from GitHub and built with Maven. Instructions to build and
test SystemML can be found in the GitHub README file.

<br/>

# Standalone vs Distributed Execution Mode

SystemML's standalone mode is designed to allow data scientists to rapidly prototype algorithms
on a single machine. The standalone release packages all required libraries into a single distribution file.
In standalone mode, all operations occur on a single node in a non-Hadoop environment. Standalone mode
is not appropriate for large datasets.

For large-scale production environments, SystemML algorithm execution can be
distributed across a multi-node cluster using [Apache Hadoop](https://hadoop.apache.org/) 
or [Apache Spark](http://spark.apache.org/). 
We will make use of standalone mode throughout this tutorial.

The examples described in `docs/SystemML_Algorithms_Reference.pdf` are written 
primarily for the hadoop distributed environment. To run those examples in
standalone mode, modify the commands by replacing "`hadoop jar SystemML.jar -f ...`" 
with "`./runStandaloneSystemML.sh ...`" on Mac/UNIX or 
with "`./runStandaloneSystemML.bat ...`" on Windows.

<br/>

# Contents of the SystemML Standalone Package

To follow along with this guide, first build a standalone package of SystemML 
{% comment %} TODO: Where to download a packaged release of SystemML, ...standalone jar 

    $ wget http://spark.tc/system-ml/system-ml-LATEST-standalone.tar.gz -O - | tar -xz -C ~/systemml-tutorial


or build it from source {% endcomment %} using [Apache Maven](http://maven.apache.org)
and unpack it to your working directory, i.e. ```~/systemml-tutorial```.

    $ git clone https://github.com/apache/incubator-systemml.git
    $ mvn clean package
    $ tar -xzf `find . -name 'system-ml*standalone.tar.gz'` -C ~/systemml-tutorial

The extracted package should have these contents:

    $ ls -lF ~/systemml-tutorial
    total 56
    drwxr-xr-x  23   algorithms/
    drwxr-xr-x   5   docs/
    drwxr-xr-x  35   lib/
    -rw-r--r--   1   log4j.properties
    -rw-r--r--   1   readme.txt
    -rwxr-xr-x   1   runStandaloneSystemML.bat*
    -rwxr-xr-x   1   runStandaloneSystemML.sh*
    -rw-r--r--   1   SystemML-config.xml

Refer to `docs/SystemML_Algorithms_Reference.pdf` for more information 
about each algorithm included in the `algorithms` folder.

For the rest of the tutorial we switch our working directory to ```~/systemml-tutorial```.

    $ cd  ~/systemml-tutorial

<br/>

# Choosing Test Data

In this tutorial we will use the [Haberman's Survival Data Set](http://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)
which can be downloaded in CSV format from the [Center for Machine Learning and Intelligent Systems](http://cml.ics.uci.edu/)

    $ wget -P data/ http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data

The [Haberman Data Set](http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.names)
has 306 instances and 4 attributes (including the class attribute):

 1. Age of patient at time of operation (numerical)
 2. Patient's year of operation (year - 1900, numerical)
 3. Number of positive axillary nodes detected (numerical)
 4. Survival status (class attribute)
   * `1` = the patient survived 5 years or longer
   * `2` = the patient died within 5 year


We will need to create a metadata file (MTD) which stores metadata information 
about the content of the data file. The name of the MTD file associated with the
data file `<filename>` must be `<filename>.mtd`. 

    $ echo '{"rows": 306, "cols": 4, "format": "csv"}' > data/haberman.data.mtd

<br/>

# Example 1 - Univariate Statistics

Let's start with a simple example, computing certain [univariate statistics](algorithms-descriptive-statistics.html#univariate-statistics)
for each feature column using the algorithm `Univar-Stats.dml` which requires 3
[arguments](algorithms-descriptive-statistics.html#arguments):

* `X`:  location of the input data file to analyze
* `TYPES`:  location of the file that contains the feature column types encoded by integer numbers: `1` = scale, `2` = nominal, `3` = ordinal
* `STATS`:  location of the output matrix of computed statistics will be stored

We need to create a file `types.csv` that describes the type of each column in 
the data along with it's metadata file `types.csv.mtd`.

    $ echo '1,1,1,2' > data/types.csv
    $ echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd
    
 
To run the `Univar-Stats.dml` algorithm, issue the following command:

    $ ./runStandaloneSystemML.sh algorithms/Univar-Stats.dml -nvargs X=data/haberman.data TYPES=data/types.csv STATS=data/univarOut.mtx

The resulting matrix has one row per each univariate statistic and one column 
per input feature. The output file `univarOut.mtx` describes that 
matrix. The elements of the first column denote the number of the statistic, 
the elements of the second column refer to the number of the feature column in 
the input data, and the elements of the third column show the value of the
univariate statistic.

    1 1 30.0
    1 2 58.0
    2 1 83.0
    2 2 69.0
    2 3 52.0
    3 1 53.0
    3 2 11.0
    3 3 52.0
    4 1 52.45751633986928
    4 2 62.85294117647059
    4 3 4.026143790849673
    5 1 116.71458266366658
    5 2 10.558630665380907
    5 3 51.691117539912135
    6 1 10.803452349303281
    6 2 3.2494046632238507
    6 3 7.189653506248555
    7 1 0.6175922641866753
    7 2 0.18575610076612029
    7 3 0.41100513466216837
    8 1 0.20594669940735139
    8 2 0.051698529971741194
    8 3 1.7857418611299172
    9 1 0.1450718616532357
    9 2 0.07798443581479181
    9 3 2.954633471088322
    10 1 -0.6150152487211726
    10 2 -1.1324380182967442
    10 3 11.425776549251449
    11 1 0.13934809593495995
    11 2 0.13934809593495995
    11 3 0.13934809593495995
    12 1 0.277810485320835
    12 2 0.277810485320835
    12 3 0.277810485320835
    13 1 52.0
    13 2 63.0
    13 3 1.0
    14 1 52.16013071895425
    14 2 62.80392156862745
    14 3 1.2483660130718954
    15 4 2.0
    16 4 1.0
    17 4 1.0

The following table lists the number and name of each univariate statistic. The row
numbers below correspond to the elements of the first column in the output 
matrix above. The signs "+" show applicability to scale or/and to categorical 
features.

  | Row | Name of Statistic          | Scale | Categ. |
  | :-: |:-------------------------- |:-----:| :-----:|
  |  1  | Minimum                    |   +   |        |
  |  2  | Maximum                    |   +   |        |
  |  3  | Range                      |   +   |        |
  |  4  | Mean                       |   +   |        |
  |  5  | Variance                   |   +   |        |
  |  6  | Standard deviation         |   +   |        |
  |  7  | Standard error of mean     |   +   |        |
  |  8  | Coefficient of variation   |   +   |        |
  |  9  | Skewness                   |   +   |        |
  | 10  | Kurtosis                   |   +   |        |
  | 11  | Standard error of skewness |   +   |        |
  | 12  | Standard error of kurtosis |   +   |        |
  | 13  | Median                     |   +   |        |
  | 14  | Inter quartile mean        |   +   |        |
  | 15  | Number of categories       |       |    +   |
  | 16  | Mode                       |       |    +   |
  | 17  | Number of modes            |       |    +   |


<br/>
<br/>

# Example 2 - Binary-class Support Vector Machines

Let's take the same `haberman.data` to explore the 
[binary-class support vector machines](algorithms-classification.html#binary-class-support-vector-machines) algorithm `l2-svm.dml`. 
This example also illustrates how to use of the sampling algorithm `sample.dml`
and the data split algorithm `spliXY.dml`.

## Sampling the Test Data

First we need to use the `sample.dml` algorithm to separate the input into one
training data set and one data set for model prediction. 

Parameters:

 * `X`       : (input)  input data set: filename of input data set
 * `sv`      : (input)  sampling vector: filename of 1-column vector w/ percentages. sum(sv) must be 1.
 * `O`       : (output) folder name w/ samples generated
 * `ofmt`    : (output) format of O: "csv", "binary" (default) 


We will create the file `perc.csv` and `perc.csv.mtd` to define the sampling vector with a sampling rate of 
50% to generate 2 data sets: 

    $ printf "0.5\n0.5" > data/perc.csv
    $ echo '{"rows": 2, "cols": 1, "format": "csv"}' > data/perc.csv.mtd

Let's run the sampling algorithm to create the two data samples:

    $ ./runStandaloneSystemML.sh algorithms/utils/sample.dml -nvargs X=data/haberman.data sv=data/perc.csv O=data/haberman.part ofmt="csv"



## Splitting Labels from Features

Next we use the `splitXY.dml` algorithm to separate the feature columns from 
the label column(s).

Parameters:

 * `X`       : (input)  filename of data matrix
 * `y`       : (input)  colIndex: starting index is 1
 * `OX`      : (output) filename of output matrix with all columns except y
 * `OY`      : (output) filename of output matrix with y column
 * `ofmt`    : (output) format of OX and OY output matrix: "csv", "binary" (default)

We specify `y=4` as the 4th column contains the labels to be predicted and run
the `splitXY.dml` algorithm on our training and test data sets.

    $ ./runStandaloneSystemML.sh algorithms/utils/splitXY.dml -nvargs X=data/haberman.part/1 y=4 OX=data/haberman.train.data.csv OY=data/haberman.train.labels.csv ofmt="csv"

    $ ./runStandaloneSystemML.sh algorithms/utils/splitXY.dml -nvargs X=data/haberman.part/2 y=4 OX=data/haberman.test.data.csv  OY=data/haberman.test.labels.csv  ofmt="csv"

## Training and Testing the Model

Now we need to train our model using the `l2-svm.dml` algorithm.

[Parameters](algorithms-classification.html#arguments-1):

 * `X`         : (input)  filename of training data features
 * `Y`         : (input)  filename of training data labels
 * `model`     : (output) filename of model that contains the learnt weights
 * `fmt`       : (output) format of model: "csv", "text" (sparse-matrix) 
 * `Log`       : (output) log file for metrics and progress while training
 * `confusion` : (output) filename of confusion matrix computed using a held-out test set (optional)

The `l2-svm.dml` algorithm is used on our training data sample to train the model.

    $ ./runStandaloneSystemML.sh algorithms/l2-svm.dml -nvargs X=data/haberman.train.data.csv Y=data/haberman.train.labels.csv model=data/l2-svm-model.csv fmt="csv" Log=data/l2-svm-log.csv

The `l2-svm-predict.dml` algorithm is used on our test data sample to predict the labels based on the trained model.

    $ ./runStandaloneSystemML.sh algorithms/l2-svm-predict.dml -nvargs X=data/haberman.test.data.csv Y=data/haberman.test.labels.csv model=data/l2-svm-model.csv fmt="csv" confusion=data/l2-svm-confusion.csv

The console output should show the accuracy of the trained model in percent, i.e.:

    15/09/01 01:32:51 INFO api.DMLScript: BEGIN DML run 09/01/2015 01:32:51
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating localtmpdir with value /tmp/systemml
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating scratch with value scratch_space
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating optlevel with value 2
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating numreducers with value 10
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating jvmreuse with value false
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating defaultblocksize with value 1000
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating dml.yarn.appmaster with value false
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating dml.yarn.appmaster.mem with value 2048
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating dml.yarn.mapreduce.mem with value 2048
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating dml.yarn.app.queue with value default
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating cp.parallel.matrixmult with value true
    15/09/01 01:32:51 INFO conf.DMLConfig: Updating cp.parallel.textio with value true
    Accuracy (%): 74.14965986394557
    15/09/01 01:32:52 INFO api.DMLScript: SystemML Statistics:
    Total execution time:		0.130 sec.
    Number of executed MR Jobs:	0.


The generated file `l2-svm-confusion.csv` should contain the following confusion matrix of this form:

    |t1 t2|
    |t3 t4|

 * The model correctly predicted label 1 `t1` times
 * The model incorrectly predicted label 1 as opposed to label 2 `t2` times
 * The model incorrectly predicted label 2 as opposed to label 1 `t3` times
 * The model correctly predicted label 2 `t4` times.

If the confusion matrix looks like this ..

    107.0,38.0
    0.0,2.0

... then the accuracy of the model is (t1+t4)/(t1+t2+t3+t4) = (107+2)/107+38+0+2) = 0.741496599

<br/>

Refer to the [Algorithms Reference](algorithms-reference.html) for more details.

<br/>

# Troubleshooting

If you encounter a `"java.lang.OutOfMemoryError"` you can edit the invocation 
script (`runStandaloneSystemML.sh` or `runStandaloneSystemML.bat`) to increase
the memory available to the JVM, i.e: 

    java -Xmx16g -Xms4g -Xmn1g -cp ${CLASSPATH} org.apache.sysml.api.DMLScript \
         -f ${SCRIPT_FILE} -exec singlenode -config=SystemML-config.xml \
         $@

<br/>

# Next Steps

Check out the [Algorithms Reference](algorithms-reference.html) and run
some of the other pre-packaged algorithms.

Follow the [DML and PyDML Programming Guide](dml-and-pydml-programming-guide.html) to become familiar with DML and PyDML.

