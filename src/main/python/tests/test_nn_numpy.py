#!/usr/bin/python
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

# Assumption: pip install keras
# 
# This test validates SystemML's deep learning APIs (Keras2DML, Caffe2DML and nn layer) by comparing the results with that of keras.
#
# To run:
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-memory 10g --driver-class-path SystemML.jar,systemml-*-extra.jar test_nn_numpy.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-memory 10g --driver-class-path SystemML.jar,systemml-*-extra.jar test_nn_numpy.py`

# Make the `systemml` package importable
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest

import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
from systemml.mllearn import Keras2DML
from pyspark.sql import SparkSession

batch_size = 32
input_shape = (3,64,64)
K.set_image_data_format("channels_first")
# K.set_image_dim_ordering("th")
keras_tensor = np.random.rand(batch_size,input_shape[0], input_shape[1], input_shape[2])
sysml_matrix = keras_tensor.reshape((batch_size, -1))
tmp_dir = 'tmp_dir'

spark = SparkSession.builder.getOrCreate()

def are_predictions_all_close(keras_model):
    sysml_model = Keras2DML(spark, keras_model, input_shape=input_shape, weights=tmp_dir)
    keras_preds = keras_model.predict(keras_tensor).flatten()
    sysml_preds = sysml_model.predict_proba(sysml_matrix).flatten()
    #print(str(keras_preds))
    #print(str(sysml_preds))
    return np.allclose(keras_preds, sysml_preds)

class TestNNLibrary(unittest.TestCase):
    def test_1layer_cnn_predictions(self):
        keras_model = Sequential()
        keras_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='valid'))
        keras_model.add(Flatten())
        keras_model.add(Dense(10, activation='softmax'))
        self.failUnless(are_predictions_all_close(keras_model))

    def test_multilayer_cnn_predictions(self):
        keras_model = Sequential()
        keras_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='valid'))
        keras_model.add(MaxPooling2D(pool_size=(2, 2)))
        keras_model.add(Conv2D(64, (3, 3), activation='relu'))
        keras_model.add(MaxPooling2D(pool_size=(2, 2)))
        keras_model.add(Flatten())
        keras_model.add(Dense(256, activation='softmax'))
        keras_model.add(Dropout(0.25))
        keras_model.add(Dense(10, activation='softmax'))
        self.failUnless(are_predictions_all_close(keras_model))    

if __name__ == '__main__':
    unittest.main()
