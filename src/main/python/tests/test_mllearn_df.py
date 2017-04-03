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

# To run:
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-class-path SystemML.jar test_mllearn_df.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-class-path SystemML.jar test_mllearn_df.py`

# Make the `systemml` package importable
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest

import numpy as np
from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
from sklearn import datasets, metrics, neighbors
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import accuracy_score, r2_score
from systemml.mllearn import LinearRegression, LogisticRegression, NaiveBayes, SVM

sc = SparkContext()
sparkSession = SparkSession.builder.getOrCreate()

# Currently not integrated with JUnit test
# ~/spark-1.6.1-scala-2.11/bin/spark-submit --master local[*] --driver-class-path SystemML.jar test.py
class TestMLLearn(unittest.TestCase):

    def test_logistic_sk2(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:int(.9 * n_samples)]
        y_train = y_digits[:int(.9 * n_samples)]
        X_test = X_digits[int(.9 * n_samples):]
        y_test = y_digits[int(.9 * n_samples):]
        # Convert to DataFrame for i/o: current way to transfer data
        logistic = LogisticRegression(sparkSession, transferUsingDF=True)
        logistic.fit(X_train, y_train)
        mllearn_predicted = logistic.predict(X_test)
        sklearn_logistic = linear_model.LogisticRegression()
        sklearn_logistic.fit(X_train, y_train)
        self.failUnless(accuracy_score(sklearn_logistic.predict(X_test), mllearn_predicted) > 0.95) # We are comparable to a similar algorithm in scikit learn

    def test_linear_regression(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sparkSession, solver='direct-solve', transferUsingDF=True)
        regr.fit(diabetes_X_train, diabetes_y_train)
        mllearn_predicted = regr.predict(diabetes_X_test)
        sklearn_regr = linear_model.LinearRegression()
        sklearn_regr.fit(diabetes_X_train, diabetes_y_train)
        self.failUnless(r2_score(sklearn_regr.predict(diabetes_X_test), mllearn_predicted) > 0.95) # We are comparable to a similar algorithm in scikit learn

    def test_linear_regression_cg(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sparkSession, solver='newton-cg', transferUsingDF=True)
        regr.fit(diabetes_X_train, diabetes_y_train)
        mllearn_predicted = regr.predict(diabetes_X_test)
        sklearn_regr = linear_model.LinearRegression()
        sklearn_regr.fit(diabetes_X_train, diabetes_y_train)
        self.failUnless(r2_score(sklearn_regr.predict(diabetes_X_test), mllearn_predicted) > 0.95) # We are comparable to a similar algorithm in scikit learn


    def test_svm_sk2(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:int(.9 * n_samples)]
        y_train = y_digits[:int(.9 * n_samples)]
        X_test = X_digits[int(.9 * n_samples):]
        y_test = y_digits[int(.9 * n_samples):]
        svm = SVM(sparkSession, is_multi_class=True, transferUsingDF=True)
        mllearn_predicted = svm.fit(X_train, y_train).predict(X_test)
        from sklearn import linear_model, svm
        clf = svm.LinearSVC()
        sklearn_predicted = clf.fit(X_train, y_train).predict(X_test)
        self.failUnless(accuracy_score(sklearn_predicted, mllearn_predicted) > 0.95 )


if __name__ == '__main__':
    unittest.main()
