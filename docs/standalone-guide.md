---
layout: global
title: SystemML Standalone Guide
description: SystemML Standalone Guide
displayTitle: SystemML Standalone Guide
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
running existing SystemML algorithms in standalone mode.


# What is SystemML

SystemML enables large-scale machine learning (ML) via a high-level declarative
language with R-like syntax called [DML](dml-language-reference.html) and
Python-like syntax called PyDML. DML and PyDML allow data scientists to
express their ML algorithms with full flexibility but without the need to fine-tune
distributed runtime execution plans and system configurations.
These ML programs are dynamically compiled and optimized based on data
and cluster characteristics using rule-based and cost-based optimization techniques.
The compiler automatically generates hybrid runtime execution plans ranging
from in-memory, single node execution to distributed computation for Hadoop
or Spark Batch execution.
SystemML features a suite of algorithms for Descriptive Statistics, Classification,
Clustering, Regression, Matrix Factorization, and Survival Analysis. Detailed descriptions of these
algorithms can be found in the [Algorithms Reference](algorithms-reference.html).

# Download SystemML

Apache SystemML releases are available from the [Downloads](http://systemml.apache.org/download.html) page.

SystemML can also be downloaded from GitHub and built with Maven.
The SystemML project is available on GitHub at [https://github.com/apache/systemml](https://github.com/apache/systemml).
Instructions to build SystemML can be found in the <a href="engine-dev-guide.html">Engine Developer Guide</a>.

# Standalone vs Distributed Execution Mode

SystemML's standalone mode is designed to allow data scientists to rapidly prototype algorithms
on a single machine. In standalone mode, all operations occur on a single node in a non-Hadoop
environment. Standalone mode is not appropriate for large datasets.

For large-scale production environments, SystemML algorithm execution can be
distributed across multi-node clusters using [Apache Hadoop](https://hadoop.apache.org/)
or [Apache Spark](http://spark.apache.org/).
We will make use of standalone mode throughout this tutorial.

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


---

# Example 1 - Univariate Statistics

Let's start with a simple example, computing certain [univariate statistics](algorithms-descriptive-statistics.html#univariate-statistics)
for each feature column using the algorithm `Univar-Stats.dml` which requires 3
[arguments](algorithms-descriptive-statistics.html#arguments):

* `X`:  location of the input data file to analyze
* `TYPES`:  location of the file that contains the feature column types encoded by integer numbers: `1` = scale, `2` = nominal, `3` = ordinal
* `STATS`:  location where the output matrix of computed statistics is to be stored

We need to create a file `types.csv` that describes the type of each column in
the data along with its metadata file `types.csv.mtd`.

    $ echo '1,1,1,2' > data/types.csv
    $ echo '{"rows": 1, "cols": 4, "format": "csv"}' > data/types.csv.mtd


To run the `Univar-Stats.dml` algorithm, issue the following command (we set the optional argument `CONSOLE_OUTPUT` to `TRUE` to print the statistics to the console):

    $ ./runStandaloneSystemML.sh scripts/algorithms/Univar-Stats.dml -nvargs X=data/haberman.data TYPES=data/types.csv STATS=data/univarOut.mtx CONSOLE_OUTPUT=TRUE

    [...]
    -------------------------------------------------
    Feature [1]: Scale
     (01) Minimum             | 30.0
     (02) Maximum             | 83.0
     (03) Range               | 53.0
     (04) Mean                | 52.45751633986928
     (05) Variance            | 116.71458266366658
     (06) Std deviation       | 10.803452349303281
     (07) Std err of mean     | 0.6175922641866753
     (08) Coeff of variation  | 0.20594669940735139
     (09) Skewness            | 0.1450718616532357
     (10) Kurtosis            | -0.6150152487211726
     (11) Std err of skewness | 0.13934809593495995
     (12) Std err of kurtosis | 0.277810485320835
     (13) Median              | 52.0
     (14) Interquartile mean  | 52.16013071895425
    -------------------------------------------------
    Feature [2]: Scale
     (01) Minimum             | 58.0
     (02) Maximum             | 69.0
     (03) Range               | 11.0
     (04) Mean                | 62.85294117647059
     (05) Variance            | 10.558630665380907
     (06) Std deviation       | 3.2494046632238507
     (07) Std err of mean     | 0.18575610076612029
     (08) Coeff of variation  | 0.051698529971741194
     (09) Skewness            | 0.07798443581479181
     (10) Kurtosis            | -1.1324380182967442
     (11) Std err of skewness | 0.13934809593495995
     (12) Std err of kurtosis | 0.277810485320835
     (13) Median              | 63.0
     (14) Interquartile mean  | 62.80392156862745
    -------------------------------------------------
    Feature [3]: Scale
     (01) Minimum             | 0.0
     (02) Maximum             | 52.0
     (03) Range               | 52.0
     (04) Mean                | 4.026143790849673
     (05) Variance            | 51.691117539912135
     (06) Std deviation       | 7.189653506248555
     (07) Std err of mean     | 0.41100513466216837
     (08) Coeff of variation  | 1.7857418611299172
     (09) Skewness            | 2.954633471088322
     (10) Kurtosis            | 11.425776549251449
     (11) Std err of skewness | 0.13934809593495995
     (12) Std err of kurtosis | 0.277810485320835
     (13) Median              | 1.0
     (14) Interquartile mean  | 1.2483660130718954
    -------------------------------------------------
    Feature [4]: Categorical (Nominal)
     (15) Num of categories   | 2
     (16) Mode                | 1
     (17) Num of modes        | 1


In addition to writing statistics to the console, the `Univar-Stats.dml` script writes the computed statistics
to the `data/univarOut.mtx` file specified by the STATS input parameter.

**univarOut.mtx file**

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


---

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

    $ ./runStandaloneSystemML.sh scripts/utils/sample.dml -nvargs X=data/haberman.data sv=data/perc.csv O=data/haberman.part ofmt="csv"


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

    $ ./runStandaloneSystemML.sh scripts/utils/splitXY.dml -nvargs X=data/haberman.part/1 y=4 OX=data/haberman.train.data.csv OY=data/haberman.train.labels.csv ofmt="csv"

    $ ./runStandaloneSystemML.sh scripts/utils/splitXY.dml -nvargs X=data/haberman.part/2 y=4 OX=data/haberman.test.data.csv  OY=data/haberman.test.labels.csv  ofmt="csv"

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

    $ ./runStandaloneSystemML.sh scripts/algorithms/l2-svm.dml -nvargs X=data/haberman.train.data.csv Y=data/haberman.train.labels.csv model=data/l2-svm-model.csv fmt="csv" Log=data/l2-svm-log.csv

The `l2-svm-predict.dml` algorithm is used on our test data sample to predict the labels based on the trained model.

    $ ./runStandaloneSystemML.sh scripts/algorithms/l2-svm-predict.dml -nvargs X=data/haberman.test.data.csv Y=data/haberman.test.labels.csv model=data/l2-svm-model.csv fmt="csv" confusion=data/l2-svm-confusion.csv

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

    |0   1.0 2.0|
    |1.0 t1  t2 |
    |2.0 t3  t4 |

 * The model correctly predicted label 1 `t1` times
 * The model incorrectly predicted label 1 as opposed to label 2 `t2` times
 * The model incorrectly predicted label 2 as opposed to label 1 `t3` times
 * The model correctly predicted label 2 `t4` times.

If the confusion matrix looks like this ...

    0,1.0,2.0
    1.0,107.0,38.0
    2.0,0.0,2.0

... then the accuracy of the model is (t1+t4)/(t1+t2+t3+t4) = (107+2)/107+38+0+2) = 0.741496599

<br/>

Refer to the [Algorithms Reference](algorithms-reference.html) for more details.


---

# Example 3 - Linear Regression

For this example, we'll use a standalone wrapper executable, `bin/systemml`, that is available to
be run directly within the project's source directory when built locally.

After you build SystemML from source (`mvn clean package`), the standalone mode can be executed
either on Linux or OS X using the `./bin/systemml` script, or on Windows using the
`.\bin\systemml.bat` batch file.

If you run from the script from the project root folder `./` or from the `./bin` folder, then the
output files from running SystemML will be created inside the `./temp` folder to keep them separate
from the SystemML source files managed by Git. The output files for this example will be created
under the `./temp` folder.

The runtime behavior and logging behavior of SystemML can be customized by editing the files
`./conf/SystemML-config.xml` and `./conf/log4j.properties`. Both files will be created from their
corresponding `*.template` files during the first execution of the SystemML executable script.

When invoking the `./bin/systemml` or `.\bin\systemml.bat` with any of the prepackaged DML scripts
you can omit the relative path to the DML script file. The following two commands are equivalent:

    ./bin/systemml ./scripts/datagen/genLinearRegressionData.dml -nvargs numSamples=1000 numFeatures=50 maxFeatureValue=5 maxWeight=5 addNoise=FALSE b=0 sparsity=0.7 output=linRegData.csv format=csv perc=0.5

    ./bin/systemml genLinearRegressionData.dml -nvargs numSamples=1000 numFeatures=50 maxFeatureValue=5 maxWeight=5 addNoise=FALSE b=0 sparsity=0.7 output=linRegData.csv format=csv perc=0.5

In this guide we invoke the command with the relative folder to make it easier to look up the source
of the DML scripts.

## Linear Regression Example

As an example of the capabilities and power of SystemML and DML, let's consider the Linear Regression algorithm.
We require sets of data to train and test our model. To obtain this data, we can either use real data or
generate data for our algorithm. The
[UCI Machine Learning Repository Datasets](https://archive.ics.uci.edu/ml/datasets.html) is one location for real data.
Use of real data typically involves some degree of data wrangling. In the following example, we will use SystemML to
generate random data to train and test our model.

This example consists of the following parts:

  * [Run DML Script to Generate Random Data](#run-dml-script-to-generate-random-data)
  * [Divide Generated Data into Two Sample Groups](#divide-generated-data-into-two-sample-groups)
  * [Split Label Column from First Sample](#split-label-column-from-first-sample)
  * [Split Label Column from Second Sample](#split-label-column-from-second-sample)
  * [Train Model on First Sample](#train-model-on-first-sample)
  * [Test Model on Second Sample](#test-model-on-second-sample)

SystemML is distributed in several packages, including a standalone package. We'll operate in Standalone mode in this
example.

<a name="run-dml-script-to-generate-random-data" />

### Run DML Script to Generate Random Data

We can execute the `genLinearRegressionData.dml` script in Standalone mode using either the `systemml` or `systemml.bat`
file.
In this example, we'll generate a matrix of 1000 rows of 50 columns of test data, with sparsity 0.7. In addition to
this, a 51<sup>st</sup> column consisting of labels will
be appended to the matrix.

    ./bin/systemml ./scripts/datagen/genLinearRegressionData.dml -nvargs numSamples=1000 numFeatures=50 maxFeatureValue=5 maxWeight=5 addNoise=FALSE b=0 sparsity=0.7 output=linRegData.csv format=csv perc=0.5

This generates the following files inside the `./temp` folder:

    linRegData.csv      # 1000 rows of 51 columns of doubles (50 data columns and 1 label column), csv format
    linRegData.csv.mtd  # Metadata file
    perc.csv            # Used to generate two subsets of the data (for training and testing)
    perc.csv.mtd        # Metadata file
    scratch_space       # SystemML scratch_space directory

<a name="divide-generated-data-into-two-sample-groups" />

### Divide Generated Data into Two Sample Groups

Next, we'll create two subsets of the generated data, each of size ~50%. We can accomplish this using the `sample.dml`
script with the `perc.csv` file created in the previous step:

    0.5
    0.5


The `sample.dml` script will randomly sample rows from the `linRegData.csv` file and place them into 2 files based
on the percentages specified in `perc.csv`. This will create two sample groups of roughly 50 percent each.

    ./bin/systemml ./scripts/utils/sample.dml -nvargs X=linRegData.csv sv=perc.csv O=linRegDataParts ofmt=csv


This script creates two partitions of the original data and places them in a `linRegDataParts` folder. The files created
are as follows:

    linRegDataParts/1       # first partition of data, ~50% of rows of linRegData.csv, csv format
    linRegDataParts/1.mtd   # metadata
    linRegDataParts/2       # second partition of data, ~50% of rows of linRegData.csv, csv format
    linRegDataParts/2.mtd   # metadata


The `1` file contains the first partition of data, and the `2` file contains the second partition of data.
An associated metadata file describes
the nature of each partition of data. If we open `1` and `2` and look at the number of rows, we can see that typically
the partitions are not exactly 50% but instead are close to 50%. However, we find that the total number of rows in the
original data file equals the sum of the number of rows in `1` and `2`.


<a name="split-label-column-from-first-sample" />

### Split Label Column from First Sample

The next task is to split the label column from the first sample. We can do this using the `splitXY.dml` script.

    ./bin/systemml ./scripts/utils/splitXY.dml -nvargs X=linRegDataParts/1 y=51 OX=linRegData.train.data.csv OY=linRegData.train.labels.csv ofmt=csv

This splits column 51, the label column, off from the data. When done, the following files have been created.

    linRegData.train.data.csv        # training data of 50 columns, csv format
    linRegData.train.data.csv.mtd    # metadata
    linRegData.train.labels.csv      # training labels of 1 column, csv format
    linRegData.train.labels.csv.mtd  # metadata


<a name="split-label-column-from-second-sample" />

### Split Label Column from Second Sample

We also need to split the label column from the second sample.

    ./bin/systemml ./scripts/utils/splitXY.dml -nvargs X=linRegDataParts/2 y=51 OX=linRegData.test.data.csv OY=linRegData.test.labels.csv ofmt=csv

This splits column 51 off the data, resulting in the following files:

    linRegData.test.data.csv        # test data of 50 columns, csv format
    linRegData.test.data.csv.mtd    # metadata
    linRegData.test.labels.csv      # test labels of 1 column, csv format
    linRegData.test.labels.csv.mtd  # metadata


<a name="train-model-on-first-sample" />

### Train Model on First Sample

Now, we can train our model based on the first sample. To do this, we utilize the `LinearRegDS.dml` (Linear Regression
Direct Solve) script. Note that SystemML also includes a `LinearRegCG.dml` (Linear Regression Conjugate Gradient)
algorithm for situations where the number of features is large.

    ./bin/systemml ./scripts/algorithms/LinearRegDS.dml -nvargs X=linRegData.train.data.csv Y=linRegData.train.labels.csv B=betas.csv fmt=csv

This will generate the following files:

    betas.csv      # betas, 50 rows of 1 column, csv format
    betas.csv.mtd  # metadata

The LinearRegDS.dml script generates statistics to standard output similar to the following.

	BEGIN LINEAR REGRESSION SCRIPT
	Reading X and Y...
	Calling the Direct Solver...
	Computing the statistics...
	AVG_TOT_Y,-2.160284487670675
	STDEV_TOT_Y,66.86434576808432
	AVG_RES_Y,-3.3127468704080085E-10
	STDEV_RES_Y,1.7231785003947183E-8
	DISPERSION,2.963950542926297E-16
	R2,1.0
	ADJUSTED_R2,1.0
	R2_NOBIAS,1.0
	ADJUSTED_R2_NOBIAS,1.0
	R2_VS_0,1.0
	ADJUSTED_R2_VS_0,1.0
	Writing the output matrix...
	END LINEAR REGRESSION SCRIPT

Now that we have our `betas.csv`, we can test our model with our second set of data.


<a name="test-model-on-second-sample" />

### Test Model on Second Sample

To test our model on the second sample, we can use the `GLM-predict.dml` script. This script can be used for both
prediction and scoring. Here, we're using it for scoring since we include the `Y` named argument. Our `betas.csv`
file is specified as the `B` named argument.

    ./bin/systemml ./scripts/algorithms/GLM-predict.dml -nvargs X=linRegData.test.data.csv Y=linRegData.test.labels.csv B=betas.csv fmt=csv

This generates statistics similar to the following to standard output.

	LOGLHOOD_Z,,FALSE,NaN
	LOGLHOOD_Z_PVAL,,FALSE,NaN
	PEARSON_X2,,FALSE,1.895530994504798E-13
	PEARSON_X2_BY_DF,,FALSE,4.202951207327712E-16
	PEARSON_X2_PVAL,,FALSE,1.0
	DEVIANCE_G2,,FALSE,0.0
	DEVIANCE_G2_BY_DF,,FALSE,0.0
	DEVIANCE_G2_PVAL,,FALSE,1.0
	LOGLHOOD_Z,,TRUE,NaN
	LOGLHOOD_Z_PVAL,,TRUE,NaN
	PEARSON_X2,,TRUE,1.895530994504798E-13
	PEARSON_X2_BY_DF,,TRUE,4.202951207327712E-16
	PEARSON_X2_PVAL,,TRUE,1.0
	DEVIANCE_G2,,TRUE,0.0
	DEVIANCE_G2_BY_DF,,TRUE,0.0
	DEVIANCE_G2_PVAL,,TRUE,1.0
	AVG_TOT_Y,1,,1.0069397725436522
	STDEV_TOT_Y,1,,68.29092137526905
	AVG_RES_Y,1,,-4.1450397073455047E-10
	STDEV_RES_Y,1,,2.0519206226041048E-8
	PRED_STDEV_RES,1,TRUE,1.0
	R2,1,,1.0
	ADJUSTED_R2,1,,1.0
	R2_NOBIAS,1,,1.0
	ADJUSTED_R2_NOBIAS,1,,1.0


We see that the STDEV_RES_Y value of the testing phase is of similar magnitude
to the value obtained from the model training phase.

For convenience, we can encapsulate our DML invocations in a single script:

	#!/bin/bash

	./bin/systemml ./scripts/datagen/genLinearRegressionData.dml -nvargs numSamples=1000 numFeatures=50 maxFeatureValue=5 maxWeight=5 addNoise=FALSE b=0 sparsity=0.7 output=linRegData.csv format=csv perc=0.5

	./bin/systemml ./scripts/utils/sample.dml -nvargs X=linRegData.csv sv=perc.csv O=linRegDataParts ofmt=csv

	./bin/systemml ./scripts/utils/splitXY.dml -nvargs X=linRegDataParts/1 y=51 OX=linRegData.train.data.csv OY=linRegData.train.labels.csv ofmt=csv

	./bin/systemml ./scripts/utils/splitXY.dml -nvargs X=linRegDataParts/2 y=51 OX=linRegData.test.data.csv OY=linRegData.test.labels.csv ofmt=csv

	./bin/systemml ./scripts/algorithms/LinearRegDS.dml -nvargs X=linRegData.train.data.csv Y=linRegData.train.labels.csv B=betas.csv fmt=csv

	./bin/systemml ./scripts/algorithms/GLM-predict.dml -nvargs X=linRegData.test.data.csv Y=linRegData.test.labels.csv B=betas.csv fmt=csv


# Troubleshooting

If you encounter a `"java.lang.OutOfMemoryError"` you can edit the invocation
script (`runStandaloneSystemML.sh` or `runStandaloneSystemML.bat`) to increase
the memory available to the JVM, i.e:

    java -Xmx16g -Xms4g -Xmn1g -cp ${CLASSPATH} org.apache.sysml.api.DMLScript \
         -f ${SCRIPT_FILE} -exec singlenode -config SystemML-config.xml \
         $@
