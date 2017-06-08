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

# Predicting Breast Cancer Proliferation Scores with Apache Spark and Apache SystemML

Note: This project is still a **work in progress**.

## Overview
The [Tumor Proliferation Assessment Challenge 2016 (TUPAC16)](http://tupac.tue-image.nl/) is a "Grand Challenge" that was created for the [2016 Medical Image Computing and Computer Assisted Intervention (MICCAI 2016)](http://miccai2016.org/en/) conference.  In this challenge, the goal is to develop state-of-the-art algorithms for automatic prediction of tumor proliferation scores from whole-slide histopathology images of breast tumors.

## Background
Breast cancer is the leading cause of cancerous death in women in less-developed countries, and is the second leading cause of cancerous deaths in developed countries, accounting for 29% of all cancers in women within the U.S. [1]. Survival rates increase as early detection increases, giving incentive for pathologists and the medical world at large to develop improved methods for even earlier detection [2].  There are many forms of breast cancer including Ductal Carcinoma in Situ (DCIS), Invasive Ductal Carcinoma (IDC), Tubular Carcinoma of the Breast, Medullary Carcinoma of the Breast, Invasive Lobular Carcinoma, Inflammatory Breast Cancer and several others [3]. Within all of these forms of breast cancer, the rate in which breast cancer cells grow (proliferation), is a strong indicator of a patient’s prognosis. Although there are many means of determining the presence of breast cancer, tumor proliferation speed has been proven to help pathologists determine the treatment for the patient. The most common technique for determining the proliferation speed is through mitotic count (mitotic index) estimates, in which a pathologist counts the dividing cell nuclei in hematoxylin and eosin (H&E) stained slide preparations to determine the number of mitotic bodies.  Given this, the pathologist produces a proliferation score of either 1, 2, or 3, ranging from better to worse prognosis [4]. Unfortunately, this approach is known to have reproducibility problems due to the variability in counting, as well as the difficulty in distinguishing between different grades.

References:
[1] http://emedicine.medscape.com/article/1947145-overview#a3
[2] http://emedicine.medscape.com/article/1947145-overview#a7
[3] http://emedicine.medscape.com/article/1954658-overview
[4] http://emedicine.medscape.com/article/1947145-workup#c12

## Goal & Approach
In an effort to automate the process of classification, this project aims to develop a large-scale deep learning approach for predicting tumor scores directly from the pixels of whole-slide histopathology images.  Our proposed approach is based on a recent research paper from Stanford [1].  Starting with 500 extremely high-resolution tumor slide images with accompanying score labels, we aim to make use of Apache Spark in a preprocessing step to cut and filter the images into smaller square samples, generating 4.7 million samples for a total of ~7TB of data [2].  We then utilize Apache SystemML on top of Spark to develop and train a custom, large-scale, deep convolutional neural network on these samples, making use of the familiar linear algebra syntax and automatically-distributed execution of SystemML [3].  Our model takes as input the pixel values of the individual samples, and is trained to predict the correct tumor score classification for each one.  In addition to distributed linear algebra, we aim to exploit task-parallelism via parallel for-loops for hyperparameter optimization, as well as hardware acceleration for faster training via a GPU-backed runtime.  Ultimately, we aim to develop a model that is sufficiently stronger than existing approaches for the task of breast cancer tumor proliferation score classification.

References:
[1] https://web.stanford.edu/group/rubinlab/pubs/2243353.pdf
[2] See [`Preprocessing.ipynb`](Preprocessing.ipynb), and [`breastcancer/preprocessing.py`](breastcancer/preprocessing.py).
[3] See [`MachineLearning.ipynb`](MachineLearning.ipynb), [`softmax_clf.dml`](softmax_clf.dml), and [`convnet.dml`](convnet.dml).

![Approach](https://apache.github.io/systemml/img/projects/breast_cancer/approach.svg)

---

## Setup (*All nodes* unless other specified):
* System Packages:
  * `sudo yum update`
  * `sudo yum install gcc ruby`
* Python 3:
  * `sudo yum install epel-release`
  * `sudo yum install -y https://centos7.iuscommunity.org/ius-release.rpm`
  * `sudo yum install -y python35u python35u-libs python35u-devel python35u-pip`
  * `ln -s /usr/bin/python3.5 ~/.local/bin/python3`
  * `ln -s /usr/bin/pip3.5 ~/.local/bin/pip3`
  * Prepend `~/.local/bin` to the `PATH`.
* OpenSlide:
  * `sudo yum install openslide`
* Python packages:
  * `pip3 install -U matplotlib numpy pandas scipy jupyter ipython scikit-learn scikit-image flask openslide-python`
* SystemML (only driver):
  * `git clone https://github.com/apache/systemml.git`
  * `cd systemml`
  * `mvn clean package`
  * `pip3 install -e src/main/python`
* Add the following to the `data` folder (same location on *all* nodes):
  * `training_image_data` folder with the training slides.
  * `testing_image_data` folder with the testing slides.
  * `training_ground_truth.csv` file containing the tumor & molecular scores for each slide.
* Layout:
  ```
  - MachineLearning.ipynb
  - Preprocessing.ipynb
  - breastcancer/
    - preprocessing.py
    - visualization.py
  - convnet.dml
  - nn/
  - ...
  - data/
    - training_ground_truth.csv
    - training_image_data
        - TUPAC-TR-001.svs
        - TUPAC-TR-002.svs
        - ...
    - testing_image_data
        - TUPAC-TE-001.svs
        - TUPAC-TE-002.svs
        - ...
  ```


* Adjust the Spark settings in `$SPARK_HOME/conf/spark-defaults.conf` using the following examples, depending on the job being executed:
  * All jobs:
    ```
    # Use most of the driver memory.
    spark.driver.memory 70g
    # Remove the max result size constraint.
    spark.driver.maxResultSize 0
    # Increase the message size.
    spark.rpc.message.maxSize 128
    # Extend the network timeout threshold.
    spark.network.timeout 1000s
    # Setup some extra Java options for performance.
    spark.driver.extraJavaOptions -server -Xmn12G
    spark.executor.extraJavaOptions -server -Xmn12G
    # Setup local directories on separate disks for intermediate read/write performance, if running
    # on Spark Standalone clusters.
    spark.local.dirs /disk2/local,/disk3/local,/disk4/local,/disk5/local,/disk6/local,/disk7/local,/disk8/local,/disk9/local,/disk10/local,/disk11/local,/disk12/local
    ```

  * Preprocessing:
    ```
    # Save 1/2 executor memory for Python processes
    spark.executor.memory 50g
    ```

  * Machine Learning:
    ```
    # Use all executor memory for JVM
    spark.executor.memory 100g
    ```

* `cd` to this `breast_cancer` folder.
* Start Jupyter + PySpark with the following command (could also use Yarn in client mode with `--master yarn --deploy-mode`):
  ```
  PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook" pyspark --master spark://MASTER_URL:7077 --driver-class-path $SYSTEMML_HOME/target/SystemML.jar --jars $SYSTEMML_HOME/target/SystemML.jar
  ```

## Create a Histopath slide “lab” to view the slides (just driver):
  - `git clone https://github.com/openslide/openslide-python.git`
  - Host locally:
    - `python3 path/to/openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 path/to/data/`
  - Host on server:
    - `python3 path/to/openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 -l HOSTING_URL_HERE path/to/data/`
    - Open local browser to `HOSTING_URL_HERE:5000`.

