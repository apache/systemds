---
layout: global
title: Beginner's Guide for Caffe2DML users
description: Beginner's Guide for Caffe2DML users
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

Caffe2DML is an experimental API that converts an Caffe specification to DML.

## Example: Train Lenet

1. Install `mlextend` package to get MNIST data: `pip install mlxtend`.
2. (Optional but recommended) Follow the steps mentioned in [the user guide]([the user guide of native backend](http://apache.github.io/incubator-systemml/native-backend)) and install Intel MKL.
3. Install [SystemML](http://apache.github.io/incubator-systemml/beginners-guide-python#install-systemml).
4. Invoke PySpark shell: `pyspark --conf spark.executorEnv.LD_LIBRARY_PATH=/path/to/blas-n-other-dependencies`.

```bash
# Download the MNIST dataset
from mlxtend.data import mnist_data
import numpy as np
from sklearn.utils import shuffle
X, y = mnist_data()
X, y = shuffle(X, y)

# Split the data into training and test
n_samples = len(X)
X_train = X[:int(.9 * n_samples)]
y_train = y[:int(.9 * n_samples)]
X_test = X[int(.9 * n_samples):]
y_test = y[int(.9 * n_samples):]

# Download the Lenet network
import urllib
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/lenet/mnist/lenet.proto', 'lenet.proto')
urllib.urlretrieve('https://raw.githubusercontent.com/niketanpansare/model_zoo/master/caffe/vision/lenet/mnist/lenet_solver.proto', 'lenet_solver.proto')

# Train Lenet On MNIST using scikit-learn like API
from systemml.mllearn import Caffe2DML
lenet = Caffe2DML(sqlCtx, solver='lenet_solver.proto', input_shape=(1, 28, 28)).set(debug=True).setStatistics(True)
lenet.fit(X_train, y_train)
y_predicted = lenet.predict(X_test)
```

## Frequently asked questions

- How to set batch size ?

Batch size is set in `data_param` of the Data layer:

	layer {
	  name: "mnist"
	  type: "Data"
	  top: "data"
	  top: "label"
	  data_param {
	    source: "mnist_train"
	    batch_size: 64
	    backend: LMDB
	  }
	}
	
- How to set maximum number of iterations for training ?

Caffe allows you to set the maximum number of iterations in solver specification

	# The maximum number of iterations
	max_iter: 2000
	
- How to set the size of the validation dataset ?

The size of the validation dataset is determined by the parameters `test_iter` and the batch size. For example: If the batch size is 64 and 
`test_iter` is 10, then the validation size is 640. This setting generates following DML code internally:

	num_images = nrow(y_full)
	BATCH_SIZE = 64
	num_validation = 10 * BATCH_SIZE
	X = X_full[(num_validation+1):num_images,]; y = y_full[(num_validation+1):num_images,]
	X_val = X_full[1:num_validation,]; y_val = y_full[1:num_validation,]
	num_images = nrow(y) 

- How to monitor loss via command-line ?

To monitor loss, please set following parameters in the solver specification

	# Display training loss and accuracy every 100 iterations
	display: 100
	# Carry out validation every 500 training iterations and display validation loss and accuracy.
	test_iter: 10
	test_interval: 500
	
 - How to pass a single jpeg image to Caffe2DML for prediction ?
 
	from PIL import Image
	import systemml as sml
	from systemml.mllearn import Caffe2DML
	img_shape = (3, 224, 224)
	input_image = sml.convertImageToNumPyArr(Image.open(img_file_path), img_shape=img_shape)
	resnet = Caffe2DML(sqlCtx, solver='ResNet_50_solver.proto', weights='ResNet_50_pretrained_weights', input_shape=img_shape)
	resnet.predict(input_image)

- How to prepare a directory of jpeg images for training with Caffe2DML ?

The below example assumes that the input dataset has 2 labels `cat` and `dogs` and the filename has these labels as prefix.
We iterate through the directory and convert each jpeg image into pyspark.ml.linalg.Vector using pyspark.
These vectors are stored as DataFrame and randomized using Spark SQL's `orderBy(rand())` function.
The DataFrame is then saved in parquet format to reduce the cost of preprocessing for repeated training.

	from systemml.mllearn import Caffe2DML
	from pyspark.sql import SQLContext
	import numpy as np
	import urllib, os, scipy.ndimage
	from pyspark.ml.linalg import Vectors
	from pyspark import StorageLevel
	import systemml as sml
	from pyspark.sql.functions import rand 
	# ImageNet specific parameters
	img_shape = (3, 224, 224)
	train_dir = '/home/biuser/dogs_vs_cats/train'
	def getLabelFeatures(filename):
		from PIL import Image
		vec = Vectors.dense(sml.convertImageToNumPyArr(Image.open(os.path.join(train_dir, filename)), img_shape=img_shape)[0,:])
		if filename.lower().startswith('cat'):
			return (1, vec)
		elif filename.lower().startswith('dog'):
			return (2, vec)
		else:
			raise ValueError('Expected the filename to start with either cat or dog')
	
	list_jpeg_files = os.listdir(train_dir)
	# 10 files per partition
	train_df = sc.parallelize(list_jpeg_files, int(len(list_jpeg_files)/10)).map(lambda filename : getLabelFeatures(filename)).toDF(['label', 'features']).orderBy(rand())
	# Optional: but helps seperates conversion-related from training
	# Alternatively, this dataframe can be passed directly to `caffe2dml_model.fit(train_df)`
	train_df.write.parquet('kaggle-cats-dogs.parquet')