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


# Apache SystemML

**Documentation:** [SystemML Documentation](http://apache.github.io/systemml/)
**Mailing List:** [Dev Mailing List](mailto:dev@systemml.apache.org)
**Build Status:** [![Build Status](https://sparktc.ibmcloud.com/jenkins/job/SystemML-DailyTest/badge/icon)](https://sparktc.ibmcloud.com/jenkins/job/SystemML-DailyTest)
**Issue Tracker:** [JIRA](https://issues.apache.org/jira/browse/SYSTEMML)
**Download:** [Download SystemML](http://systemml.apache.org/download.html)

**SystemML** is now an **Apache Top Level Project**! Please see the [**Apache SystemML**](http://systemml.apache.org/)
website for more information.

SystemML is a flexible, scalable machine learning system.
SystemML's distinguishing characteristics are:

  1. **Algorithm customizability via R-like and Python-like languages**.
  2. **Multiple execution modes**, including Spark MLContext API, Spark Batch, Hadoop Batch, Standalone, and JMLC.
  3. **Automatic optimization** based on data and cluster characteristics to ensure both efficiency and scalability.


## Algorithm Customizability

ML algorithms in SystemML are specified in a high-level, declarative machine learning (DML) language.
Algorithms can be expressed in either an R-like syntax or a Python-like syntax. DML includes
linear algebra primitives, statistical functions, and additional constructs.

This high-level language significantly increases the productivity of
data scientists as it provides (1) full flexibility in expressing custom
analytics and (2) data independence from the underlying input formats and
physical data representations.


## Multiple Execution Modes

SystemML computations can be executed in a variety of different modes. To begin with, SystemML
can be operated in Standalone mode on a single machine, allowing data scientists to develop
algorithms locally without need of a distributed cluster. In order to scale up, algorithms can also be distributed
across a cluster using Spark or Hadoop.
This flexibility allows the utilization of an organization's existing resources and expertise.
In addition, SystemML features a
[Spark MLContext API](http://apache.github.io/systemml/spark-mlcontext-programming-guide.html)
that allows for programmatic interaction via Scala, Python, and Java. SystemML also features an
embedded API for scoring models.


## Automatic Optimization

Algorithms specified in DML are dynamically compiled and optimized based on data and cluster characteristics
using rule-based and cost-based optimization techniques. The optimizer automatically generates hybrid runtime
execution plans ranging from in-memory, single-node execution, to distributed computations on Spark or Hadoop.
This ensures both efficiency and scalability. Automatic optimization reduces or eliminates the need to hand-tune
distributed runtime execution plans and system configurations.

## ML Algorithms

SystemML features a suite of production-level examples that can be grouped into six broad categories:
Descriptive Statistics, Classification, Clustering, Regression, Matrix Factorization, and Survival Analysis.
Detailed descriptions of these algorithms can be found in the
[SystemML Algorithms Reference](http://apache.github.io/systemml/algorithms-reference.html).  The goal of these provided algorithms is to serve as production-level examples that can modified or used as inspiration for a new custom algorithm.

## Download & Setup

Before you get started on SystemML, make sure that your environment is set up and ready to go.

  1. **If you’re on OS X, we recommend installing [Homebrew](http://brew.sh) if you haven’t already.  For Linux users, the [Linuxbrew project](http://linuxbrew.sh/) is equivalent.**

  OS X:
  ```
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  ```
  Linux:
  ```
  ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"
  ```

  2. **Install Java (need Java 8).**
  ```
  brew tap caskroom/cask
  brew install Caskroom/cask/java
  ```

  3. **Install Spark 2.1.**
  ```
  brew tap homebrew/versions
  brew install apache-spark21
  ```

  4. **Download SystemML.**

  Go to the [SystemML Downloads page](http://systemml.apache.org/download.html), download `systemml-0.15.0.zip` (should be 2nd), and unzip it to a location of your choice.

  *The next step is optional, but it will make your life a lot easier.*

  5. **[OPTIONAL] Set `SYSTEMML_HOME` in your bash profile.**
  Add the following to `~/.bash_profile`, replacing `path/to/` with the location of the download in step 5.
  ```
  export SYSTEMML_HOME=path/to/systemml-0.15.0
  ```
  *Make sure to open a new tab in terminal so that you make sure the changes have been made.*

  6. **[OPTIONAL] Install Python or Python 3 (to follow along with our Jupyter notebook examples).**

  Python 2:
  ```
  brew install python
  pip install jupyter matplotlib numpy
  ```

  Python 3:
  ```
  brew install python3
  pip3 install jupyter matplotlib numpy
  ```

**Congrats! You can now use SystemML!**

## Next Steps!

To get started, please consult the
[SystemML Documentation](http://apache.github.io/systemml/) website on GitHub.  We
recommend using the [Spark MLContext API](http://apache.github.io/systemml/spark-mlcontext-programming-guide.html)
to run SystemML from Scala or Python using `spark-shell`, `pyspark`, or `spark-submit`.
