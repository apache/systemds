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

# SystemML-NN Examples

#### This folder contains scripts and PySpark Jupyter notebooks serving as examples of using the *SystemML-NN* (`nn`) deep learning library.

---

# Examples
### MNIST Softmax Classifier

* This example trains a softmax classifier, which is essentially a multi-class logistic regression model, on the MNIST data.  The model will be trained on the *training* images, validated on the *validation* images, and tested for final performance metrics on the *test* images.
* Notebook: `Example - MNIST Softmax Classifier.ipynb`.
* DML Functions: `mnist_softmax.dml`
* Training script: `mnist_softmax-train.dml`
* Prediction script: `mnist_softmax-predict.dml`

### MNIST "LeNet" Neural Net

* This example trains a neural network on the MNIST data using a ["LeNet" architecture](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). The model will be trained on the *training* images, validated on the *validation* images, and tested for final performance metrics on the *test* images.
* Notebook: `Example - MNIST LeNet.ipynb`.
* DML Functions: `mnist_lenet.dml`
* Training script: `mnist_lenet-train.dml`
* Prediction script: `mnist_lenet-predict.dml`

---

# Setup
## Code
* To run the examples, please first download and unzip the project via GitHub using the "Clone or download" button on the [homepage of the project](https://github.com/dusenberrymw/systemml-nn), *or* via the following commands:

  ```
  curl -LO https://github.com/dusenberrymw/systemml-nn/archive/master.zip
  unzip master.zip
  ```

* Then, move into the `examples` folder via:
  ```
  cd systemml-nn-master/examples/
  ```

## Data
* These examples use the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which contains labeled 28x28 pixel images of handwritten digits in the range of 0-9.  There are 60,000 training images, and 10,000 testing images.  Of the 60,000 training images, 5,000 will be used as validation images.
* **Download**:
  * **Notebooks**: The data will be automatically downloaded as a step in either of the example notebooks.
  * **Training scripts**: Please run `get_mnist_data.sh` to download the data separately.

## Execution
* These examples contain scripts written in SystemML's R-like language (`*.dml`), as well as PySpark Jupyter notebooks (`*.ipynb`).  The scripts contain the math for the algorithms, enclosed in functions, and the notebooks serve as full, end-to-end examples of reading in data, training models using the functions within the scripts, and evaluating final performance.
* **Notebooks**: To run the notebook examples, please install the SystemML Python package with `pip install systemml`, and then startup Jupyter in the following manner from this directory (or for more information, please see [this great blog post](http://spark.tc/0-to-life-changing-application-with-apache-systemml/)):

  ```
  PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook" pyspark --master local[*] --driver-memory 3G --driver-class-path $SYSTEMML_HOME/SystemML.jar --jars $SYSTEMML_HOME/SystemML.jar
  ```

  Note that all printed output, such as training statistics, from the SystemML scripts will be sent to the terminal in which Jupyter was started (for now...).

* **Scripts**: To run the scripts from the command line using `spark-submit`, please see the comments located at the top of the `-train` and `-predict` scripts.
