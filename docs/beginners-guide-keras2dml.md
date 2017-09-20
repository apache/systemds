---
layout: global
title: Beginner's Guide for Keras2DML users
description: Beginner's Guide for Keras2DML users
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

Keras2DML is an **experimental API** that converts a Keras specification to DML through the intermediate Caffe2DML module. 
It is designed to fit well into the mllearn framework and hence supports NumPy, Pandas as well as PySpark DataFrame.

### Getting Started 

To create a Keras2DML object, one needs to create a Keras model through the Funcitonal API. please see the [Functional API.](https://keras.io/models/model/)
This module utilizes the existing [Caffe2DML](beginners-guide-caffe2dml) backend to convert Keras models into DML. Keras models are 
parsed and translated into Caffe prototext and caffemodel files which are then piped into Caffe2DML. Thus one can follow the Caffe2DML
documentation for further information.

### Model Conversion

Keras models are parsed based on their layer structure and corresponding weights and translated into the relative Caffe layer and weight
configuration. Be aware that currently this is a translation into Caffe and there will be loss of information from keras models such as 
intializer information, and other layers which do not exist in Caffe. 

To create a Keras2DML object, simply pass the keras object to the Keras2DML constructor. It's also important to note that your models
should be compiled so that the loss can be accessed for Caffe2DML

```python
from systemml.mllearn import Keras2DML
import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

model = ResNet50(weights='imagenet',include_top=True,pooling='None',input_shape=(224,224,3))
model.compile(optimizer='sgd', loss= 'categorical_crossentropy')

resnet = Keras2DML(spark,model,input_shape=(3,224,224))
resnet.summary()
```

