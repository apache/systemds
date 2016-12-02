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
from pyspark.sql import SQLContext
from sklearn import datasets, metrics, neighbors
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from systemml.mllearn import LinearRegression, LogisticRegression, NaiveBayes, SVM

sc = SparkContext()
sqlCtx = SQLContext(sc)

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
        logistic = LogisticRegression(sqlCtx, transferUsingDF=True)
        score = logistic.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.9)

    def test_linear_regression_sk2(self):
        diabetes = datasets.load_diabetes()
        diabetes_X = diabetes.data[:, np.newaxis, 2]
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        regr = LinearRegression(sqlCtx, transferUsingDF=True)
        regr.fit(diabetes_X_train, diabetes_y_train)
        score = regr.score(diabetes_X_test, diabetes_y_test)
        self.failUnless(score > 0.4) # TODO: Improve r2-score (may be I am using it incorrectly)

    def test_svm_sk2(self):
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = len(X_digits)
        X_train = X_digits[:int(.9 * n_samples)]
        y_train = y_digits[:int(.9 * n_samples)]
        X_test = X_digits[int(.9 * n_samples):]
        y_test = y_digits[int(.9 * n_samples):]
        svm = SVM(sqlCtx, is_multi_class=True, transferUsingDF=True)
        score = svm.fit(X_train, y_train).score(X_test, y_test)
        self.failUnless(score > 0.9)

    #def test_naive_bayes_sk2(self):
    #    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    #    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    #    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    #    vectorizer = TfidfVectorizer()
    #    # Both vectors and vectors_test are SciPy CSR matrix
    #    vectors = vectorizer.fit_transform(newsgroups_train.data)
    #    vectors_test = vectorizer.transform(newsgroups_test.data)
    #    nb = NaiveBayes(sqlCtx)
    #    nb.fit(vectors, newsgroups_train.target)
    #    pred = nb.predict(vectors_test)
    #    score = metrics.f1_score(newsgroups_test.target, pred, average='weighted')
    #    self.failUnless(score > 0.8)


if __name__ == '__main__':
    unittest.main()
