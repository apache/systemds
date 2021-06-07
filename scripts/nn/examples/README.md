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

# SystemDS-NN Examples

## MNIST Softmax Classifier

* This example trains a softmax classifier, which is essentially a multi-class logistic regression model, on the MNIST data.
The model will be trained on the *training* images, validated on the *validation* images, and tested for final performance metrics on the *test* images.
* DML Functions: `mnist_softmax.dml`
* Training script: `mnist_softmax-train.dml`
* Prediction script: `mnist_softmax-predict.dml`

## MNIST "LeNet" Neural Net

* This example trains a neural network on the MNIST data using a ["LeNet" architecture](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
The model will be trained on the *training* images, validated on the *validation* images, and tested for final performance metrics on the *test* images.
* DML Functions: `mnist_lenet.dml`
* Training script: `mnist_lenet-train.dml`
* Prediction script: `mnist_lenet-predict.dml`

### Neural Collaborative Filtering

* This example trains a neural network on the MovieLens data set using the concept of [Neural Collaborative Filtering (NCF)](https://dl.acm.org/doi/abs/10.1145/3038912.3052569)
that is aimed at approaching recommendation problems using deep neural networks as opposed to common matrix factorization approaches.
* As in the original paper, the targets are binary and only indicate whether a user has rated a movie or not.
This makes the recommendation problem harder than working with the values of the ratings, but interaction data is in practice easier to collect.
* MovieLens only provides positive interactions in form of ratings. We therefore randomly sample negative interactions as suggested by the original paper.
* The implementation works with a fixed layer architecture with two embedding layers at the beginning for users and items,
three dense layers with ReLu activations in the middle and a sigmoid activation for the final classification.
