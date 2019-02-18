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
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-memory 10g  --driver-class-path ../../../../target/SystemML.jar,../../../../target/systemml-*-extra.jar test_nn_numpy.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-memory 10g --driver-class-path SystemML.jar,systemml-*-extra.jar test_nn_numpy.py`

# Make the `systemml` package importable
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest

import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, UpSampling2D, SimpleRNN, Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from systemml.mllearn import Keras2DML
from pyspark.sql import SparkSession
from pyspark import SparkContext
from keras.utils import np_utils
from scipy import stats
from sklearn.preprocessing import normalize
from operator import mul

batch_size = 32
K.set_image_data_format('channels_first')
# K.set_image_dim_ordering("th")

def get_tensor(shape, random=True):
    if shape[0] is None:
        # Use the first dimension is None, use batch size:
        shape = list(shape)
        shape[0] = batch_size
    return (np.random.randint(100, size=shape) + 1) / 100

tmp_dir = 'tmp_dir'

sc  = SparkContext()
spark = SparkSession.builder.getOrCreate()
sc.setLogLevel('ERROR')

def initialize_weights(model):
    for l in range(len(model.layers)):
        if model.layers[l].get_weights() is not None or len(model.layers[l].get_weights()) > 0:
            model.layers[l].set_weights([get_tensor(elem.shape) for elem in model.layers[l].get_weights()])
    return model

def get_input_output_shape(layers):
    tmp_keras_model = Sequential()
    for layer in layers:
        tmp_keras_model.add(layer)
    return tmp_keras_model.layers[0].input_shape, tmp_keras_model.layers[-1].output_shape

def get_one_hot_encoded_labels(output_shape):
    output_cells = reduce(mul, list(output_shape[1:]), 1)
    y = np.array(np.random.choice(output_cells, batch_size))
    y[0] = output_cells - 1
    one_hot_labels = np_utils.to_categorical(y, num_classes=output_cells)
    return one_hot_labels

def get_sysml_model(keras_model):
    sysml_model = Keras2DML(spark, keras_model, weights=tmp_dir, max_iter=1, batch_size=batch_size)
    # For apples-to-apples comparison of output probabilities:
    # By performing one-hot encoding outside, we ensure that the ordering of the TF columns
    # matches that of SystemML
    sysml_model.set(train_algo='batch', perform_one_hot_encoding=False)
    # print('Script:' + str(sysml_model.get_training_script()))
    return sysml_model

def base_test(layers, add_dense=False, test_backward=True):
    layers = [layers] if not isinstance(layers, list) else layers
    in_shape, output_shape = get_input_output_shape(layers)
    # --------------------------------------
    # Create Keras model
    keras_model = Sequential()
    for layer in layers:
        keras_model.add(layer)
    if len(output_shape) > 2:
        # Flatten the last layer activation before feeding it to the softmax loss
        keras_model.add(Flatten())
    if add_dense:
        keras_model.add(Dense(num_labels, activation='softmax'))
    else:
        keras_model.add(Activation('softmax'))
    keras_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, decay=0, momentum=0, nesterov=False))
    # --------------------------------------
    keras_model = initialize_weights(keras_model)
    sysml_model = get_sysml_model(keras_model)
    keras_tensor = get_tensor(in_shape)
    sysml_matrix = keras_tensor.reshape((batch_size, -1))
    #if len(keras_tensor.shape) == 4:
    #    keras_tensor = np.flip(keras_tensor, 1)
    # --------------------------------------
    sysml_preds = sysml_model.predict_proba(sysml_matrix)
    if test_backward:
        one_hot_labels = get_one_hot_encoded_labels(keras_model.layers[-1].output_shape)
        sysml_model.fit(sysml_matrix, one_hot_labels)
        sysml_preds = sysml_model.predict_proba(sysml_matrix)
    keras_preds = keras_model.predict(keras_tensor)
    if test_backward:
        keras_model.train_on_batch(keras_tensor, one_hot_labels)
        keras_preds = keras_model.predict(keras_tensor)
    # --------------------------------------
    if len(output_shape) == 4:
        # Flatten doesnot respect channel_first, so reshuffle the dimensions:
        keras_preds = keras_preds.reshape((batch_size, output_shape[2], output_shape[3], output_shape[1]))
        keras_preds = np.swapaxes(keras_preds, 2, 3)  # (h,w,c) -> (h,c,w)
        keras_preds = np.swapaxes(keras_preds, 1, 2)  # (h,c,w) -> (c,h,w)
    elif len(output_shape) > 4:
        raise Exception('Unsupported output shape:' + str(output_shape))
    # --------------------------------------
    return sysml_preds, keras_preds, keras_model, output_shape

def debug_layout(sysml_preds, keras_preds):
    for i in range(len(keras_preds.shape)):
        print('After flipping along axis=' + str(i) + ' => ' + str(np.allclose(sysml_preds, np.flip(keras_preds, i).flatten())))

def test_forward(layers):
    sysml_preds, keras_preds, keras_model, output_shape = base_test(layers, test_backward=False)
    ret = np.allclose(sysml_preds.flatten(), keras_preds.flatten())
    if not ret:
        print('The forward test failed for the model:' + str(keras_model.summary()))
        print('SystemML output:' + str(sysml_preds))
        print('Keras output:' + str(keras_preds))
        #debug_layout(sysml_preds.flatten(),
        #             keras_preds.reshape((-1, output_shape[1], output_shape[2], output_shape[3])))
    return ret

def test_backward(layers):
    sysml_preds, keras_preds, keras_model, output_shape = base_test(layers, test_backward=True)
    ret = np.allclose(sysml_preds.flatten(), keras_preds.flatten())
    if not ret:
        print('The backward test failed for the model:' + str(keras_model.summary()))
        print('SystemML output:' + str(sysml_preds))
        print('Keras output:' + str(keras_preds))
        # debug_layout(sysml_preds.flatten(),
        #             keras_preds.reshape((-1, output_shape[1], output_shape[2], output_shape[3])))
    return ret


class TestNNLibrary(unittest.TestCase):

    def test_dense_forward(self):
        self.failUnless(test_forward(Dense(10, input_shape=[30])))

    def test_dense_backward(self):
        self.failUnless(test_backward(Dense(10, input_shape=[30])))

    def test_lstm_forward1(self):
        self.failUnless(test_forward(LSTM(2, return_sequences=True, activation='tanh', stateful=False, recurrent_activation='sigmoid', input_shape=(3, 4))))

    #def test_lstm_backward1(self):
    #    self.failUnless(test_backward(LSTM(2, return_sequences=True, activation='tanh', stateful=False, recurrent_activation='sigmoid',  input_shape=(3, 4))))

    def test_lstm_forward2(self):
        self.failUnless(test_forward(LSTM(10, return_sequences=False, activation='tanh', stateful=False, recurrent_activation='sigmoid', input_shape=(30, 20))))

    def test_lstm_backward2(self):
        self.failUnless(test_backward(LSTM(10, return_sequences=False, activation='tanh', stateful=False, recurrent_activation='sigmoid',  input_shape=(30, 20))))

    def test_dense_relu_forward(self):
        self.failUnless(test_forward(Dense(10, activation='relu', input_shape=[30])))

    def test_dense_relu_backward(self):
        self.failUnless(test_backward(Dense(10, activation='relu', input_shape=[30])))

    def test_dense_sigmoid_forward(self):
        self.failUnless(test_forward(Dense(10, activation='sigmoid', input_shape=[30])))

    def test_dense_sigmoid_backward(self):
        self.failUnless(test_backward(Dense(10, activation='sigmoid', input_shape=[30])))

    def test_dense_softmax_forward(self):
        self.failUnless(test_forward(Dense(10, activation='softmax', input_shape=[30])))

    def test_dense_softmax_backward(self):
        self.failUnless(test_backward(Dense(10, activation='softmax', input_shape=[30])))

    def test_maxpool2d_forward(self):
        self.failUnless(test_forward(MaxPooling2D(pool_size=(2, 2), input_shape=(1, 64, 32))))

    def test_maxpool2d_backward(self):
        self.failUnless(test_backward(MaxPooling2D(pool_size=(2, 2), input_shape=(1, 64, 32))))

    def test_maxpool2d_multi_channel_forward(self):
        self.failUnless(test_forward(MaxPooling2D(pool_size=(2, 2), input_shape=(3, 64, 32))))

    def test_maxpool2d_multi_channel_backward(self):
        self.failUnless(test_backward(MaxPooling2D(pool_size=(2, 2), input_shape=(3, 64, 32))))

    def test_conv2d_forward_single_channel_input_output(self):
        # 1-channel input and output
        self.failUnless(
            test_forward(Conv2D(1, kernel_size=(3, 3), input_shape=(1, 64, 64), activation='relu', padding='valid')))

    def test_conv2d_forward_single_channel_input(self):
        # 1-channel input
        self.failUnless(
            test_forward(Conv2D(32, kernel_size=(3, 3), input_shape=(1, 64, 64), activation='relu', padding='valid')))

    def test_conv2d_forward_single_channel_output(self):
        # 1-channel output
        self.failUnless(
            test_forward(Conv2D(1, kernel_size=(3, 3), input_shape=(3, 64, 64), activation='relu', padding='valid')))

    def test_conv2d_forward(self):
        self.failUnless(
            test_forward(Conv2D(32, kernel_size=(3, 3), input_shape=(3, 64, 32), activation='relu', padding='valid')))

    def test_conv2d_backward_single_channel_input_output(self):
        # 1-channel input and output
        self.failUnless(
            test_backward(Conv2D(1, kernel_size=(3, 3), input_shape=(1, 64, 64), activation='relu', padding='valid')))

    def test_conv2d_backward_single_channel_input(self):
        # 1-channel input
        self.failUnless(
            test_backward(Conv2D(32, kernel_size=(3, 3), input_shape=(1, 64, 64), activation='relu', padding='valid')))

    def test_conv2d_backward_single_channel_output(self):
        # 1-channel output
        self.failUnless(
            test_backward(Conv2D(1, kernel_size=(3, 3), input_shape=(3, 64, 64), activation='relu', padding='valid')))

    def test_conv2d_backward(self):
        self.failUnless(
            test_backward(Conv2D(32, kernel_size=(3, 3), input_shape=(3, 64, 32), activation='relu', padding='valid')))

    def test_upsampling_forward(self):
        self.failUnless(test_forward(UpSampling2D(size=(2, 2), input_shape=(3, 64, 32))))

    def test_upsampling_backward(self):
        self.failUnless(test_backward(UpSampling2D(size=(2, 2), input_shape=(3, 64, 32))))

if __name__ == '__main__':
    unittest.main()
